package primitive

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"regexp"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// makeTestInput builds a deterministic 48x36 test image with structure.
func makeTestInput() *image.RGBA {
	const w, h = 48, 36
	im := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := im.PixOffset(x, y)
			im.Pix[i+0] = uint8((x * 5) % 256)
			im.Pix[i+1] = uint8((y * 7) % 256)
			im.Pix[i+2] = uint8(((x-24)*(x-24) + (y-18)*(y-18)) % 256)
			im.Pix[i+3] = 255
		}
	}
	return im
}

func runModel(t *testing.T, workers int, seed int64, steps int, shapeType ShapeType) *Model {
	t.Helper()
	model := NewModel(makeTestInput(), Color{128, 128, 128, 255}, 128, workers, seed)
	ctx := context.Background()
	for i := 0; i < steps; i++ {
		if _, err := model.Step(ctx, shapeType, 128, 0); err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
	}
	return model
}

func pngBytes(t *testing.T, im image.Image) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, im); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// TestDeterministicRepeatRuns: same input, params and seed must produce
// identical shapes, colors and output bytes on repeated runs.
func TestDeterministicRepeatRuns(t *testing.T) {
	a := runModel(t, 2, 42, 4, ShapeTypeTriangle)
	b := runModel(t, 2, 42, 4, ShapeTypeTriangle)

	if a.SVG() != b.SVG() {
		t.Fatal("determinism: SVG outputs differ between identical runs")
	}
	if !bytes.Equal(a.Current.Pix, b.Current.Pix) {
		t.Fatal("determinism: raster buffers differ between identical runs")
	}
	if !bytes.Equal(pngBytes(t, a.Context.Image()), pngBytes(t, b.Context.Image())) {
		t.Fatal("determinism: PNG bytes differ between identical runs")
	}
	if a.Score != b.Score {
		t.Fatalf("determinism: scores differ: %v vs %v", a.Score, b.Score)
	}
}

// TestWorkerCountInvariance: the final image must not depend on how many
// workers executed the search.
func TestWorkerCountInvariance(t *testing.T) {
	reference := runModel(t, 1, 7, 3, ShapeTypeEllipse)
	for _, workers := range []int{2, 3, 5, 8} {
		t.Run(fmt.Sprintf("workers=%d", workers), func(t *testing.T) {
			model := runModel(t, workers, 7, 3, ShapeTypeEllipse)
			if model.SVG() != reference.SVG() {
				t.Fatalf("worker count %d changed the shape sequence", workers)
			}
			if !bytes.Equal(model.Current.Pix, reference.Current.Pix) {
				t.Fatalf("worker count %d changed the final pixels", workers)
			}
			if model.Score != reference.Score {
				t.Fatalf("worker count %d changed the score: %v vs %v",
					workers, model.Score, reference.Score)
			}
		})
	}
}

// TestSeedChangesOutput: a different seed should produce a different shape
// sequence (sanity check that the seed actually drives the search).
func TestSeedChangesOutput(t *testing.T) {
	a := runModel(t, 2, 1, 3, ShapeTypeTriangle)
	b := runModel(t, 2, 2, 3, ShapeTypeTriangle)
	if a.SVG() == b.SVG() {
		t.Fatal("different seeds produced identical SVG; seed is not wired up")
	}
}

// TestTieBreakIsStable: with exactly equal energies the winner is decided
// by the stable shape encoding, independent of evaluation/scan order.
func TestTieBreakIsStable(t *testing.T) {
	worker := NewWorker(makeTestInput())
	mkTriangle := func(x1, y1, x2, y2, x3, y3 int) *Triangle {
		return &Triangle{worker, x1, y1, x2, y2, x3, y3}
	}
	// two distinct shapes, forced to the same energy
	shapes := []Shape{
		mkTriangle(1, 1, 10, 1, 5, 10),
		mkTriangle(2, 2, 20, 2, 9, 12),
		mkTriangle(0, 0, 3, 0, 1, 2),
	}
	// find the expected winner: smallest stateKey
	want := shapes[0]
	for _, s := range shapes[1:] {
		if stateKey(s) < stateKey(want) {
			want = s
		}
	}
	// try every permutation of arrival order; the winner must not change
	perm := [][]int{{0, 1, 2}, {2, 1, 0}, {1, 2, 0}, {2, 0, 1}, {0, 2, 1}, {1, 0, 2}}
	for _, order := range perm {
		var best Shape
		var bestKey string
		for _, i := range order {
			s := shapes[i]
			key := stateKey(s)
			if best == nil || betterCandidate(0.5, key, 0.5, bestKey) {
				best = s
				bestKey = key
			}
		}
		if stateKey(best) != stateKey(want) {
			t.Fatalf("order %v picked %q, want %q", order, stateKey(best), stateKey(want))
		}
	}
}

// cancelAfterContext returns context.Canceled from Err() after n checks.
// Unlike a timer, this makes mid-search cancellation deterministic: the
// search checks Err() before every candidate, so cancelling after n checks
// always lands inside the search regardless of machine speed.
type cancelAfterContext struct {
	context.Context
	n int32
}

func (c *cancelAfterContext) Err() error {
	if atomic.AddInt32(&c.n, -1) < 0 {
		return context.Canceled
	}
	return nil
}

// TestStepCancellation: a cancelled context stops the search promptly and
// leaves the model untouched.
func TestStepCancellation(t *testing.T) {
	model := NewModel(makeTestInput(), Color{128, 128, 128, 255}, 128, 4, 42)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled before the search starts
	start := time.Now()
	_, err := model.Step(ctx, ShapeTypeTriangle, 128, 0)
	if err == nil {
		t.Fatal("cancelled step: got nil error")
	}
	if err != context.Canceled {
		t.Fatalf("cancelled step: got %v, want context.Canceled", err)
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Fatalf("cancelled step took %v; cancellation is not prompt", elapsed)
	}
	if len(model.Shapes) != 0 {
		t.Fatalf("cancelled step added %d shapes to the model", len(model.Shapes))
	}

	// cancellation mid-search: the context cancels after 50 Err() checks,
	// which is guaranteed to be inside the candidate search
	big := NewModel(makeGradientTarget(256, 256), Color{128, 128, 128, 255}, 256, 4, 42)
	midCtx := &cancelAfterContext{Context: context.Background(), n: 50}
	start = time.Now()
	_, err = big.Step(midCtx, ShapeTypeTriangle, 128, 0)
	if err == nil {
		t.Fatal("mid-search cancel: got nil error")
	}
	if err != context.Canceled {
		t.Fatalf("mid-search cancel: got %v, want context.Canceled", err)
	}
	if elapsed := time.Since(start); elapsed > 10*time.Second {
		t.Fatalf("mid-search cancel took %v; cancellation is not prompt", elapsed)
	}
	if len(big.Shapes) != 0 {
		t.Fatalf("cancelled step added %d shapes to the model", len(big.Shapes))
	}
}

// TestSVGReplay: parse the SVG output, replay it shape-by-shape through the
// scanline rasterizer, and require pixel-exact equality with the model's
// own raster buffer. This verifies the SVG encodes the same shapes in the
// same order with the same colors.
func TestSVGReplay(t *testing.T) {
	model := runModel(t, 2, 42, 5, ShapeTypeTriangle)
	svg := model.SVG()

	size := model.Current.Bounds().Size()
	w, h := size.X, size.Y

	fillRe := regexp.MustCompile(`fill="#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})"`)
	opacityRe := regexp.MustCompile(`fill-opacity="([0-9.]+)"`)
	pointsRe := regexp.MustCompile(`points="([^"]+)"`)

	parseFill := func(line string) (int, int, int) {
		m := fillRe.FindStringSubmatch(line)
		if m == nil {
			t.Fatalf("SVG line missing fill: %s", line)
		}
		r, _ := strconv.ParseInt(m[1], 16, 0)
		g, _ := strconv.ParseInt(m[2], 16, 0)
		b, _ := strconv.ParseInt(m[3], 16, 0)
		return int(r), int(g), int(b)
	}

	replay := image.NewRGBA(image.Rect(0, 0, w, h))
	shapeCount := 0
	for _, line := range strings.Split(svg, "\n") {
		switch {
		case strings.HasPrefix(line, "<rect "):
			r, g, b := parseFill(line)
			draw.Draw(replay, replay.Bounds(), &image.Uniform{
				color.NRGBA{uint8(r), uint8(g), uint8(b), 255}}, image.Point{}, draw.Src)
		case strings.HasPrefix(line, "<polygon "):
			r, g, b := parseFill(line)
			om := opacityRe.FindStringSubmatch(line)
			if om == nil {
				t.Fatalf("SVG polygon missing fill-opacity: %s", line)
			}
			opacity, err := strconv.ParseFloat(om[1], 64)
			if err != nil {
				t.Fatalf("bad fill-opacity in %s: %v", line, err)
			}
			a := int(math.Round(opacity * 255))
			pm := pointsRe.FindStringSubmatch(line)
			if pm == nil {
				t.Fatalf("SVG polygon missing points: %s", line)
			}
			var coords []int
			for _, pair := range strings.Split(pm[1], " ") {
				var x, y int
				if _, err := fmt.Sscanf(pair, "%d,%d", &x, &y); err != nil {
					t.Fatalf("bad point %q in %s: %v", pair, line, err)
				}
				coords = append(coords, x, y)
			}
			if len(coords) != 6 {
				t.Fatalf("triangle SVG should have 3 points, got %d numbers in %s", len(coords), line)
			}
			lines := rasterizeTriangle(coords[0], coords[1], coords[2], coords[3], coords[4], coords[5], nil)
			lines = cropScanlines(lines, w, h)
			drawLines(replay, Color{r, g, b, a}, lines)
			shapeCount++
		}
	}
	if shapeCount != len(model.Shapes) {
		t.Fatalf("SVG has %d shapes, model has %d", shapeCount, len(model.Shapes))
	}
	if !bytes.Equal(replay.Pix, model.Current.Pix) {
		for i := range replay.Pix {
			if replay.Pix[i] != model.Current.Pix[i] {
				px := i / 4
				t.Fatalf("SVG replay diverges at pixel (%d,%d) channel %d: got %d, want %d",
					px%w, px/w, i%4, replay.Pix[i], model.Current.Pix[i])
			}
		}
	}
}

// TestFramesShapeOrder: animation frames must replay shapes in the same
// order as the model and end at the same image as the live context.
func TestFramesShapeOrder(t *testing.T) {
	model := runModel(t, 2, 42, 4, ShapeTypeTriangle)
	frames := model.Frames(0) // every shape gets a frame
	if len(frames) != len(model.Shapes)+1 {
		t.Fatalf("got %d frames, want %d (background + one per shape)",
			len(frames), len(model.Shapes)+1)
	}
	last := imageToRGBA(frames[len(frames)-1])
	want := imageToRGBA(model.Context.Image())
	if !bytes.Equal(last.Pix, want.Pix) {
		t.Fatal("final animation frame differs from the model's rendered image")
	}
}

func BenchmarkDifferenceFull(b *testing.B) {
	target := makeGradientTarget(256, 256)
	current := uniformRGBA(image.Rect(0, 0, 256, 256), color.NRGBA{128, 128, 128, 255})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		differenceFullTotal(target, current)
	}
}

func BenchmarkDifferencePartial(b *testing.B) {
	target := makeGradientTarget(256, 256)
	before := uniformRGBA(image.Rect(0, 0, 256, 256), color.NRGBA{128, 128, 128, 255})
	after := copyRGBA(before)
	lines := []Scanline{{Y: 100, X1: 20, X2: 200, Alpha: 0xffff}, {Y: 101, X1: 30, X2: 190, Alpha: 0xffff}}
	drawLines(after, Color{10, 20, 30, 128}, lines)
	total := differenceFullTotal(target, before)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		differencePartialTotal(target, before, after, total, lines)
	}
}

// BenchmarkModelStep is repeatable: the seed is fixed, so every benchmark
// run searches the exact same candidate sequence.
func BenchmarkModelStep(b *testing.B) {
	model := NewModel(makeTestInput(), Color{128, 128, 128, 255}, 128, 1, 42)
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := model.Step(ctx, ShapeTypeTriangle, 128, 0); err != nil {
			b.Fatal(err)
		}
	}
}
