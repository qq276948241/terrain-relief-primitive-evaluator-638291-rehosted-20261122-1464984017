package main

import (
	"context"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/fogleman/primitive/primitive"
	"github.com/nfnt/resize"
)

var (
	Input      string
	Outputs    flagArray
	Background string
	Configs    shapeConfigArray
	Alpha      int
	InputSize  int
	OutputSize int
	Mode       int
	Workers    int
	Nth        int
	Repeat     int
	Seed       int64
	V, VV      bool
)

type flagArray []string

func (i *flagArray) String() string {
	return strings.Join(*i, ", ")
}

func (i *flagArray) Set(value string) error {
	*i = append(*i, value)
	return nil
}

type shapeConfig struct {
	Count  int
	Mode   int
	Alpha  int
	Repeat int
}

type shapeConfigArray []shapeConfig

func (i *shapeConfigArray) String() string {
	return ""
}

func (i *shapeConfigArray) Set(value string) error {
	var n int64
	_, err := fmt.Sscanf(value, "%d", &n)
	if err != nil {
		return fmt.Errorf("invalid shape count %q: %v", value, err)
	}
	*i = append(*i, shapeConfig{int(n), Mode, Alpha, Repeat})
	return nil
}

func init() {
	flag.StringVar(&Input, "i", "", "input image path")
	flag.Var(&Outputs, "o", "output image path")
	flag.Var(&Configs, "n", "number of primitives")
	flag.StringVar(&Background, "bg", "", "background color (hex)")
	flag.IntVar(&Alpha, "a", 128, "alpha value")
	flag.IntVar(&InputSize, "r", 256, "resize large input images to this size")
	flag.IntVar(&OutputSize, "s", 1024, "output image size")
	flag.IntVar(&Mode, "m", 1, "0=combo 1=triangle 2=rect 3=ellipse 4=circle 5=rotatedrect 6=beziers 7=rotatedellipse 8=polygon")
	flag.IntVar(&Workers, "j", -1, "number of parallel workers (default uses all cores; must be >= 1 when set)")
	flag.IntVar(&Nth, "nth", 1, "save every Nth frame (put \"%d\" in path)")
	flag.IntVar(&Repeat, "rep", 0, "add N extra shapes per iteration with reduced search")
	flag.Int64Var(&Seed, "seed", -1, "random seed (-1 picks a random seed and prints it)")
	flag.BoolVar(&V, "v", false, "verbose")
	flag.BoolVar(&VV, "vv", false, "very verbose")
}

// pipelineParams carries everything runPipeline needs; keeping it separate
// from the flag globals makes the pipeline testable.
type pipelineParams struct {
	input      string
	outputs    []string
	configs    []shapeConfig
	background string
	inputSize  int
	outputSize int
	workers    int
	nth        int
	seed       int64
}

// pipelineResult reports what a run actually completed.
type pipelineResult struct {
	seed      int64
	frames    int
	completed []string // output paths fully written, in write order
}

// validate checks parameters and returns a locatable error naming the
// offending flag or path.
func validate(p *pipelineParams) error {
	if p.input == "" {
		return fmt.Errorf("input argument required (-i)")
	}
	if len(p.outputs) == 0 {
		return fmt.Errorf("output argument required (-o)")
	}
	if len(p.configs) == 0 {
		return fmt.Errorf("number argument required (-n)")
	}
	for _, config := range p.configs {
		if config.Count < 1 {
			return fmt.Errorf("invalid -n value %d: shape count must be > 0", config.Count)
		}
		if config.Mode < 0 || config.Mode > 8 {
			return fmt.Errorf("invalid -m value %d: mode must be between 0 and 8", config.Mode)
		}
		if config.Alpha < 0 || config.Alpha > 255 {
			return fmt.Errorf("invalid -a value %d: alpha must be between 0 and 255", config.Alpha)
		}
		if config.Repeat < 0 {
			return fmt.Errorf("invalid -rep value %d: repeat must be >= 0", config.Repeat)
		}
	}
	if p.inputSize < 0 {
		return fmt.Errorf("invalid -r value %d: input size must be >= 0", p.inputSize)
	}
	if p.outputSize < 1 {
		return fmt.Errorf("invalid -s value %d: output size must be >= 1", p.outputSize)
	}
	if p.workers == 0 {
		return fmt.Errorf("invalid -j value 0: worker count must be >= 1 (omit -j to use all cores)")
	}
	if p.workers < 0 {
		p.workers = runtime.NumCPU()
	}
	if p.nth < 1 {
		return fmt.Errorf("invalid -nth value %d: must be >= 1", p.nth)
	}
	seen := make(map[string]string)
	for _, output := range p.outputs {
		key := output
		if output != "-" {
			abs, err := filepath.Abs(output)
			if err == nil {
				key = abs
			}
		}
		if prev, ok := seen[key]; ok {
			return fmt.Errorf("duplicate output path %q (already given as %q)", output, prev)
		}
		seen[key] = output
		if output == "-" {
			continue
		}
		// Fail fast on unwritable output directories instead of running the
		// whole search first.
		dir := filepath.Dir(output)
		f, err := os.CreateTemp(dir, ".primitive-check-*")
		if err != nil {
			return fmt.Errorf("output directory %s is not writable: %w", dir, err)
		}
		f.Close()
		os.Remove(f.Name())
	}
	return nil
}

// runPipeline executes the full search and writes all requested outputs.
// It returns the result describing which outputs completed; on error or
// cancellation, completed lists every output that was fully written.
func runPipeline(ctx context.Context, p pipelineParams) (*pipelineResult, error) {
	if err := validate(&p); err != nil {
		return nil, err
	}

	// resolve the seed: -1 means "pick one at random and report it"
	seed := p.seed
	if seed < 0 {
		seed = rand.New(rand.NewSource(time.Now().UTC().UnixNano())).Int63()
	}
	primitive.Log(1, "seed=%d\n", seed)
	result := &pipelineResult{seed: seed}

	// read input image
	primitive.Log(1, "reading %s\n", p.input)
	input, err := primitive.LoadImage(p.input)
	if err != nil {
		return result, fmt.Errorf("cannot read input %s: %w", p.input, err)
	}

	// scale down input image if needed
	size := uint(p.inputSize)
	if size > 0 {
		input = resize.Thumbnail(size, size, input, resize.Bilinear)
	}

	// determine background color
	var bg primitive.Color
	if p.background == "" {
		bg = primitive.MakeColor(primitive.AverageImageColor(input))
	} else {
		bg = primitive.MakeHexColor(p.background)
	}

	// run algorithm
	model := primitive.NewModel(input, bg, p.outputSize, p.workers, seed)
	primitive.Log(1, "%d: t=%.3f, score=%.6f\n", 0, 0.0, model.Score)
	start := time.Now()
	frame := 0
	for j, config := range p.configs {
		primitive.Log(1, "count=%d, mode=%d, alpha=%d, repeat=%d\n",
			config.Count, config.Mode, config.Alpha, config.Repeat)

		for i := 0; i < config.Count; i++ {
			if err := ctx.Err(); err != nil {
				return result, err
			}
			frame++

			// find optimal shape and add it to the model
			t := time.Now()
			n, err := model.Step(ctx, primitive.ShapeType(config.Mode), config.Alpha, config.Repeat)
			if err != nil {
				return result, fmt.Errorf("search cancelled at frame %d: %w", frame, err)
			}
			nps := primitive.NumberString(float64(n) / time.Since(t).Seconds())
			elapsed := time.Since(start).Seconds()
			primitive.Log(1, "%d: t=%.3f, score=%.6f, n=%d, n/s=%s\n", frame, elapsed, model.Score, n, nps)

			// write output image(s)
			for _, output := range p.outputs {
				if err := ctx.Err(); err != nil {
					return result, err
				}
				ext := strings.ToLower(filepath.Ext(output))
				if output == "-" {
					ext = ".svg"
				}
				percent := strings.Contains(output, "%")
				saveFrames := percent && ext != ".gif"
				saveFrames = saveFrames && frame%p.nth == 0
				last := j == len(p.configs)-1 && i == config.Count-1
				if saveFrames || last {
					path := output
					if percent {
						path = fmt.Sprintf(output, frame)
					}
					primitive.Log(1, "writing %s\n", path)
					switch ext {
					default:
						return result, fmt.Errorf("unrecognized file extension for output %q: %s", output, ext)
					case ".png":
						err = primitive.SavePNG(path, model.Context.Image())
					case ".jpg", ".jpeg":
						err = primitive.SaveJPG(path, model.Context.Image(), 95)
					case ".svg":
						err = primitive.SaveFile(path, model.SVG())
					case ".gif":
						frames := model.Frames(0.001)
						err = primitive.SaveGIFImageMagick(ctx, path, frames, 50, 250)
					}
					if err != nil {
						return result, fmt.Errorf("cannot write output %s: %w", path, err)
					}
					result.completed = append(result.completed, path)
				}
			}
		}
	}
	result.frames = frame
	return result, nil
}

func errorMessage(message string) bool {
	fmt.Fprintln(os.Stderr, message)
	return false
}

func main() {
	// parse and validate arguments
	flag.Parse()
	ok := true
	if Input == "" {
		ok = errorMessage("ERROR: input argument required")
	}
	if len(Outputs) == 0 {
		ok = errorMessage("ERROR: output argument required")
	}
	if len(Configs) == 0 {
		ok = errorMessage("ERROR: number argument required")
	}
	if len(Configs) == 1 {
		Configs[0].Mode = Mode
		Configs[0].Alpha = Alpha
		Configs[0].Repeat = Repeat
	}
	if !ok {
		fmt.Println("Usage: primitive [OPTIONS] -i input -o output -n count")
		flag.PrintDefaults()
		os.Exit(1)
	}

	// set log level
	if V {
		primitive.LogLevel = 1
	}
	if VV {
		primitive.LogLevel = 2
	}

	params := pipelineParams{
		input:      Input,
		outputs:    Outputs,
		configs:    Configs,
		background: Background,
		inputSize:  InputSize,
		outputSize: OutputSize,
		workers:    Workers,
		nth:        Nth,
		seed:       Seed,
	}

	// Ctrl-C / SIGTERM cancels the search and any in-flight writes.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	result, err := runPipeline(ctx, params)
	if result != nil && result.seed >= 0 {
		fmt.Fprintf(os.Stderr, "seed: %d\n", result.seed)
	}
	if err != nil {
		if result != nil && len(result.completed) > 0 {
			fmt.Fprintf(os.Stderr, "completed outputs before stopping: %s\n",
				strings.Join(result.completed, ", "))
		}
		if ctx.Err() != nil {
			fmt.Fprintln(os.Stderr, "cancelled: no further outputs were written")
			os.Exit(130)
		}
		fmt.Fprintln(os.Stderr, "ERROR:", err)
		os.Exit(1)
	}
}
