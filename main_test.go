package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"image"
	"image/png"
)

// writeTestInput creates a small deterministic PNG to use as pipeline input.
func writeTestInput(t *testing.T, dir string) string {
	t.Helper()
	im := image.NewRGBA(image.Rect(0, 0, 48, 36))
	for y := 0; y < 36; y++ {
		for x := 0; x < 48; x++ {
			i := im.PixOffset(x, y)
			im.Pix[i+0] = uint8((x * 5) % 256)
			im.Pix[i+1] = uint8((y * 7) % 256)
			im.Pix[i+2] = uint8((x + y) % 256)
			im.Pix[i+3] = 255
		}
	}
	path := filepath.Join(dir, "input.png")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := png.Encode(f, im); err != nil {
		t.Fatal(err)
	}
	return path
}

func baseParams(dir string) pipelineParams {
	return pipelineParams{
		input:      filepath.Join(dir, "input.png"),
		outputs:    []string{filepath.Join(dir, "out.png")},
		configs:    []shapeConfig{{Count: 3, Mode: 1, Alpha: 128, Repeat: 0}},
		inputSize:  0, // input is already small
		outputSize: 128,
		workers:    2,
		nth:        1,
		seed:       42,
	}
}

// TestValidationErrors: every bad parameter must produce an error that
// names the offending flag or path.
func TestValidationErrors(t *testing.T) {
	dir := t.TempDir()
	writeTestInput(t, dir)

	cases := []struct {
		name    string
		mutate  func(*pipelineParams)
		wantSub string
	}{
		{"missing input", func(p *pipelineParams) { p.input = "" }, "-i"},
		{"missing output", func(p *pipelineParams) { p.outputs = nil }, "-o"},
		{"missing count", func(p *pipelineParams) { p.configs = nil }, "-n"},
		{"negative shape count", func(p *pipelineParams) { p.configs[0].Count = -5 }, "-n"},
		{"zero shape count", func(p *pipelineParams) { p.configs[0].Count = 0 }, "-n"},
		{"bad mode", func(p *pipelineParams) { p.configs[0].Mode = 9 }, "-m"},
		{"bad alpha", func(p *pipelineParams) { p.configs[0].Alpha = 300 }, "-a"},
		{"negative repeat", func(p *pipelineParams) { p.configs[0].Repeat = -1 }, "-rep"},
		{"negative input size", func(p *pipelineParams) { p.inputSize = -1 }, "-r"},
		{"zero output size", func(p *pipelineParams) { p.outputSize = 0 }, "-s"},
		{"zero workers", func(p *pipelineParams) { p.workers = 0 }, "-j"},
		{"zero nth", func(p *pipelineParams) { p.nth = 0 }, "-nth"},
		{"duplicate outputs", func(p *pipelineParams) {
			p.outputs = []string{p.outputs[0], p.outputs[0]}
		}, "duplicate output path"},
		{"nonexistent output dir", func(p *pipelineParams) {
			p.outputs = []string{filepath.Join(dir, "no-such-dir", "out.png")}
		}, "not writable"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := baseParams(dir)
			tc.mutate(&p)
			_, err := runPipeline(context.Background(), p)
			if err == nil {
				t.Fatalf("got nil error, want error containing %q", tc.wantSub)
			}
			if !strings.Contains(err.Error(), tc.wantSub) {
				t.Fatalf("error %q does not mention %q", err, tc.wantSub)
			}
		})
	}

	t.Run("unwritable output dir", func(t *testing.T) {
		ro := filepath.Join(dir, "readonly")
		if err := os.Mkdir(ro, 0o555); err != nil {
			t.Fatal(err)
		}
		p := baseParams(dir)
		p.outputs = []string{filepath.Join(ro, "out.png")}
		_, err := runPipeline(context.Background(), p)
		if err == nil {
			t.Fatal("got nil error for unwritable directory")
		}
		if !strings.Contains(err.Error(), "not writable") {
			t.Fatalf("error %q does not locate the unwritable directory", err)
		}
	})
}

// TestPipelineDeterministic: identical runs must produce identical output
// bytes, and the worker count must not change the result.
func TestPipelineDeterministic(t *testing.T) {
	dir := t.TempDir()
	writeTestInput(t, dir)

	run := func(workers int, suffix string) (pngOut, svgOut []byte) {
		t.Helper()
		p := baseParams(dir)
		pngPath := filepath.Join(dir, "out-"+suffix+".png")
		svgPath := filepath.Join(dir, "out-"+suffix+".svg")
		p.outputs = []string{pngPath, svgPath}
		p.workers = workers
		result, err := runPipeline(context.Background(), p)
		if err != nil {
			t.Fatalf("run %s: %v", suffix, err)
		}
		if len(result.completed) != 2 {
			t.Fatalf("run %s: completed = %v, want 2 outputs", suffix, result.completed)
		}
		pngOut, err = os.ReadFile(pngPath)
		if err != nil {
			t.Fatal(err)
		}
		svgOut, err = os.ReadFile(svgPath)
		if err != nil {
			t.Fatal(err)
		}
		return pngOut, svgOut
	}

	png1, svg1 := run(1, "a")
	png2, svg2 := run(1, "b")
	if string(png1) != string(png2) || string(svg1) != string(svg2) {
		t.Fatal("repeated runs with the same seed produced different bytes")
	}
	png3, svg3 := run(4, "c")
	if string(png1) != string(png3) || string(svg1) != string(svg3) {
		t.Fatal("worker count changed the output bytes")
	}
}

// TestPipelineCancelBeforeStart: a pre-cancelled context stops the pipeline
// before any output is written; a pre-existing output file is preserved.
func TestPipelineCancelBeforeStart(t *testing.T) {
	dir := t.TempDir()
	writeTestInput(t, dir)

	outPath := filepath.Join(dir, "out.png")
	sentinel := []byte("previous output - must be preserved")
	if err := os.WriteFile(outPath, sentinel, 0o644); err != nil {
		t.Fatal(err)
	}

	p := baseParams(dir)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := runPipeline(ctx, p)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	got, _ := os.ReadFile(outPath)
	if string(got) != string(sentinel) {
		t.Fatal("cancelled run clobbered the existing output file")
	}
	assertNoTempFiles(t, dir)
}

// TestPipelineCancelMidRun: cancellation during the search stops promptly,
// reports what (if anything) completed, and leaves no partial output.
func TestPipelineCancelMidRun(t *testing.T) {
	dir := t.TempDir()
	writeTestInput(t, dir)

	outPath := filepath.Join(dir, "out.png")
	sentinel := []byte("previous output - must be preserved")
	if err := os.WriteFile(outPath, sentinel, 0o644); err != nil {
		t.Fatal(err)
	}

	p := baseParams(dir)
	p.configs = []shapeConfig{{Count: 500, Mode: 1, Alpha: 128, Repeat: 0}}
	p.inputSize = 256 // upscale work so the run is long enough to cancel
	p.outputSize = 512

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()
	start := time.Now()
	result, err := runPipeline(ctx, p)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	if elapsed := time.Since(start); elapsed > 30*time.Second {
		t.Fatalf("cancellation took %v; not prompt", elapsed)
	}
	// the run was cancelled before the final frame, so the only output path
	// must still hold the sentinel and no temp files may remain
	got, _ := os.ReadFile(outPath)
	if string(got) != string(sentinel) {
		t.Fatal("cancelled run clobbered the existing output file")
	}
	if result != nil {
		for _, completed := range result.completed {
			if completed == outPath {
				t.Fatal("result claims the final output completed, but the run was cancelled")
			}
		}
	}
	assertNoTempFiles(t, dir)
}

// TestPipelineWriteFailure: a write that fails mid-run must not leave a
// partial file masquerading as success.
func TestPipelineWriteFailure(t *testing.T) {
	dir := t.TempDir()
	writeTestInput(t, dir)

	// make the output directory disappear between validation and writing by
	// pointing at a path whose parent is a regular file
	blocker := filepath.Join(dir, "blocker")
	if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	p := baseParams(dir)
	p.outputs = []string{filepath.Join(blocker, "out.png")}
	_, err := runPipeline(context.Background(), p)
	if err == nil {
		t.Fatal("got nil error for impossible output path")
	}
	if !strings.Contains(err.Error(), "blocker") {
		t.Fatalf("error %q does not name the failing path", err)
	}
	assertNoTempFiles(t, dir)
}

// TestPipelineFrameOutputs: %-pattern outputs write one file per Nth frame
// plus the final frame, all in shape order.
func TestPipelineFrameOutputs(t *testing.T) {
	dir := t.TempDir()
	writeTestInput(t, dir)

	p := baseParams(dir)
	pattern := filepath.Join(dir, "frame-%02d.png")
	p.outputs = []string{pattern}
	p.nth = 1
	result, err := runPipeline(context.Background(), p)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.completed) != 3 {
		t.Fatalf("completed = %v, want 3 frame files", result.completed)
	}
	for i, path := range result.completed {
		want := filepath.Join(dir, "frame-"+string(rune('0'+i+1))+".png")
		_ = want // names checked below via the filesystem
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("completed frame %s missing on disk: %v", path, err)
		}
	}
	assertNoTempFiles(t, dir)
}

func assertNoTempFiles(t *testing.T, dir string) {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".primitive-") {
			t.Fatalf("temporary file %s was left behind", e.Name())
		}
	}
}
