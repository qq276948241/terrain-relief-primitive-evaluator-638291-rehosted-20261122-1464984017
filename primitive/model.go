package primitive

import (
	"context"
	"fmt"
	"image"
	"math/rand"
	"strings"

	"github.com/fogleman/gg"
)

type Model struct {
	Sw, Sh     int
	Scale      float64
	Background Color
	Target     *image.RGBA
	Current    *image.RGBA
	Context    *gg.Context
	Score      float64
	Shapes     []Shape
	Colors     []Color
	Scores     []float64
	Workers    []*Worker

	// Seed is the root random seed. Every candidate search derives its own
	// random source from (Seed, step, job), so results depend only on the
	// seed and the requested work — never on worker count, scheduling or
	// wall-clock time.
	Seed int64
	// total is the exact integer squared-error total behind Score.
	total uint64
	// steps counts completed Step calls; it partitions the seed space.
	steps int
}

func NewModel(target image.Image, background Color, size, numWorkers int, seed int64) *Model {
	w := target.Bounds().Size().X
	h := target.Bounds().Size().Y
	aspect := float64(w) / float64(h)
	var sw, sh int
	var scale float64
	if aspect >= 1 {
		sw = size
		sh = int(float64(size) / aspect)
		scale = float64(size) / float64(w)
	} else {
		sw = int(float64(size) * aspect)
		sh = size
		scale = float64(size) / float64(h)
	}

	model := &Model{}
	model.Sw = sw
	model.Sh = sh
	model.Scale = scale
	model.Background = background
	model.Target = imageToRGBA(target)
	model.Current = uniformRGBA(target.Bounds(), background.NRGBA())
	model.total = differenceFullTotal(model.Target, model.Current)
	model.Score = totalToScore(model.total, w, h)
	model.Context = model.newContext()
	model.Seed = seed
	for i := 0; i < numWorkers; i++ {
		worker := NewWorker(model.Target)
		model.Workers = append(model.Workers, worker)
	}
	return model
}

// deriveSeed mixes (seed, step, job) into an independent random seed. The
// splitmix64 finalizer makes nearby inputs produce unrelated streams.
func deriveSeed(seed int64, step, job int) int64 {
	x := uint64(seed)
	x = x*0x9E3779B97F4A7C15 + uint64(step) + 1
	x = x*0x9E3779B97F4A7C15 + uint64(job) + 1
	x ^= x >> 30
	x *= 0xBF58476D1CE4E5B9
	x ^= x >> 27
	x *= 0x94D049BB133111EB
	x ^= x >> 31
	return int64(x)
}

// stateKey is the stable shape encoding used to break exact energy ties.
// Two shapes with the same key render identically, so the choice between
// them cannot affect the output — but it must be deterministic.
func stateKey(shape Shape) string {
	return shape.SVG("")
}

// betterCandidate reports whether (energy, key) beats (bestEnergy, bestKey)
// under the deterministic ordering: lower energy wins, exact ties are
// broken by the lexicographically smaller shape encoding.
func betterCandidate(energy float64, key string, bestEnergy float64, bestKey string) bool {
	if energy != bestEnergy {
		return energy < bestEnergy
	}
	return key < bestKey
}

func (model *Model) newContext() *gg.Context {
	dc := gg.NewContext(model.Sw, model.Sh)
	dc.Scale(model.Scale, model.Scale)
	dc.Translate(0.5, 0.5)
	dc.SetColor(model.Background.NRGBA())
	dc.Clear()
	return dc
}

func (model *Model) Frames(scoreDelta float64) []image.Image {
	var result []image.Image
	dc := model.newContext()
	result = append(result, imageToRGBA(dc.Image()))
	previous := 10.0
	for i, shape := range model.Shapes {
		c := model.Colors[i]
		dc.SetRGBA255(c.R, c.G, c.B, c.A)
		shape.Draw(dc, model.Scale)
		dc.Fill()
		score := model.Scores[i]
		delta := previous - score
		if delta >= scoreDelta {
			previous = score
			result = append(result, imageToRGBA(dc.Image()))
		}
	}
	return result
}

func (model *Model) SVG() string {
	bg := model.Background
	var lines []string
	lines = append(lines, fmt.Sprintf("<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"%d\" height=\"%d\">", model.Sw, model.Sh))
	lines = append(lines, fmt.Sprintf("<rect x=\"0\" y=\"0\" width=\"%d\" height=\"%d\" fill=\"#%02x%02x%02x\" />", model.Sw, model.Sh, bg.R, bg.G, bg.B))
	lines = append(lines, fmt.Sprintf("<g transform=\"scale(%f) translate(0.5 0.5)\">", model.Scale))
	for i, shape := range model.Shapes {
		c := model.Colors[i]
		attrs := "fill=\"#%02x%02x%02x\" fill-opacity=\"%f\""
		attrs = fmt.Sprintf(attrs, c.R, c.G, c.B, float64(c.A)/255)
		lines = append(lines, shape.SVG(attrs))
	}
	lines = append(lines, "</g>")
	lines = append(lines, "</svg>")
	return strings.Join(lines, "\n")
}

func (model *Model) Add(shape Shape, alpha int) {
	before := copyRGBA(model.Current)
	lines := shape.Rasterize()
	color := computeColor(model.Target, model.Current, lines, alpha)
	drawLines(model.Current, color, lines)
	model.total = differencePartialTotal(model.Target, before, model.Current, model.total, lines)
	size := model.Target.Bounds().Size()
	score := totalToScore(model.total, size.X, size.Y)

	model.Score = score
	model.Shapes = append(model.Shapes, shape)
	model.Colors = append(model.Colors, color)
	model.Scores = append(model.Scores, score)

	model.Context.SetRGBA255(color.R, color.G, color.B, color.A)
	shape.Draw(model.Context, model.Scale)
}

func (model *Model) Step(ctx context.Context, shapeType ShapeType, alpha, repeat int) (int, error) {
	step := model.steps
	model.steps++

	state, err := model.runWorkers(ctx, step, shapeType, alpha, 1000, 100, 16)
	if err != nil {
		return 0, err
	}
	// state = HillClimb(state, 1000).(*State)
	model.Add(state.Shape, state.Alpha)

	for i := 0; i < repeat; i++ {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		state.Worker.SetRand(rand.New(rand.NewSource(deriveSeed(model.Seed, step, 16+i))))
		state.Worker.Init(model.Current, model.total)
		a := state.Energy()
		next, err := HillClimbContext(ctx, state, 100)
		if err != nil {
			return 0, err
		}
		state = next.(*State)
		b := state.Energy()
		if a == b {
			break
		}
		model.Add(state.Shape, state.Alpha)
	}

	// for _, w := range model.Workers[1:] {
	// 	model.Workers[0].Heatmap.AddHeatmap(w.Heatmap)
	// }
	// SavePNG("heatmap.png", model.Workers[0].Heatmap.Image(0.5))

	counter := 0
	for _, worker := range model.Workers {
		counter += worker.Counter
	}
	return counter, nil
}

// runWorkers evaluates exactly m independent search jobs. Job j always uses
// the random stream derived from (Seed, step, j); jobs are dealt to workers
// round-robin, so the candidate set and every candidate's random stream are
// identical for any worker count. The winner is chosen by the deterministic
// (energy, shape key) ordering over all jobs — never by arrival order.
func (model *Model) runWorkers(ctx context.Context, step int, t ShapeType, a, n, age, m int) (*State, error) {
	wn := len(model.Workers)
	type result struct {
		job   int
		state *State
	}
	ch := make(chan result, m)
	done := make(chan struct{}, wn)
	jobs := make(chan int, m)
	for j := 0; j < m; j++ {
		jobs <- j
	}
	close(jobs)
	for i := 0; i < wn; i++ {
		worker := model.Workers[i]
		worker.Init(model.Current, model.total)
		go func(worker *Worker) {
			defer func() { done <- struct{}{} }()
			for j := range jobs {
				if ctx.Err() != nil {
					return
				}
				worker.SetRand(rand.New(rand.NewSource(deriveSeed(model.Seed, step, j))))
				state, err := worker.BestHillClimbState(ctx, t, a, n, age, 1)
				if err != nil {
					return
				}
				ch <- result{j, state}
			}
		}(worker)
	}
	var bestEnergy float64
	var bestKey string
	var bestState *State
	received := 0
	finished := 0
	for received < m && finished < wn {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-done:
			finished++
		case r := <-ch:
			received++
			energy := r.state.Energy()
			key := stateKey(r.state.Shape)
			if bestState == nil || betterCandidate(energy, key, bestEnergy, bestKey) {
				bestEnergy = energy
				bestKey = key
				bestState = r.state
			}
		}
	}
	if received < m {
		// every worker exited before all jobs reported: the search was
		// cancelled (or the context is otherwise done)
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return nil, context.Canceled
	}
	return bestState, nil
}
