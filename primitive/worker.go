package primitive

import (
	"context"
	"image"
	"math/rand"

	"github.com/golang/freetype/raster"
)

type Worker struct {
	W, H       int
	Target     *image.RGBA
	Current    *image.RGBA
	Buffer     *image.RGBA
	Rasterizer *raster.Rasterizer
	Lines      []Scanline
	Heatmap    *Heatmap
	Rnd        *rand.Rand
	Score      float64
	Total      uint64
	Counter    int
}

func NewWorker(target *image.RGBA) *Worker {
	w := target.Bounds().Size().X
	h := target.Bounds().Size().Y
	worker := Worker{}
	worker.W = w
	worker.H = h
	worker.Target = target
	worker.Buffer = image.NewRGBA(target.Bounds())
	worker.Rasterizer = raster.NewRasterizer(w, h)
	worker.Lines = make([]Scanline, 0, 4096) // TODO: based on height
	worker.Heatmap = NewHeatmap(w, h)
	// A worker never owns a long-lived random stream; the model installs a
	// per-job source with SetRand before every search job. The seed here is
	// only a deterministic placeholder.
	worker.Rnd = rand.New(rand.NewSource(0))
	return &worker
}

// SetRand installs the random source for the next job. It must only be
// called by the goroutine that currently owns the worker, so the mutable
// rand state is never shared between goroutines.
func (worker *Worker) SetRand(rnd *rand.Rand) {
	worker.Rnd = rnd
}

func (worker *Worker) Init(current *image.RGBA, total uint64) {
	worker.Current = current
	worker.Total = total
	worker.Score = totalToScore(total, worker.W, worker.H)
	worker.Counter = 0
	worker.Heatmap.Clear()
}

func (worker *Worker) Energy(shape Shape, alpha int) float64 {
	worker.Counter++
	lines := shape.Rasterize()
	// worker.Heatmap.Add(lines)
	color := computeColor(worker.Target, worker.Current, lines, alpha)
	copyLines(worker.Buffer, worker.Current, lines)
	drawLines(worker.Buffer, color, lines)
	total := differencePartialTotal(worker.Target, worker.Current, worker.Buffer, worker.Total, lines)
	return totalToScore(total, worker.W, worker.H)
}

func (worker *Worker) BestHillClimbState(ctx context.Context, t ShapeType, a, n, age, m int) (*State, error) {
	var bestEnergy float64
	var bestKey string
	var bestState *State
	for i := 0; i < m; i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		state, err := worker.BestRandomState(ctx, t, a, n)
		if err != nil {
			return nil, err
		}
		before := state.Energy()
		climbed, err := HillClimbContext(ctx, state, age)
		if err != nil {
			return nil, err
		}
		finalState := climbed.(*State)
		energy := finalState.Energy()
		vv("%dx random: %.6f -> %dx hill climb: %.6f\n", n, before, age, energy)
		if bestState == nil || betterCandidate(energy, stateKey(finalState.Shape), bestEnergy, bestKey) {
			bestEnergy = energy
			bestKey = stateKey(finalState.Shape)
			bestState = finalState
		}
	}
	return bestState, nil
}

func (worker *Worker) BestRandomState(ctx context.Context, t ShapeType, a, n int) (*State, error) {
	var bestEnergy float64
	var bestKey string
	var bestState *State
	for i := 0; i < n; i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		state := worker.RandomState(t, a)
		energy := state.Energy()
		if bestState == nil || betterCandidate(energy, stateKey(state.Shape), bestEnergy, bestKey) {
			bestEnergy = energy
			bestKey = stateKey(state.Shape)
			bestState = state
		}
	}
	return bestState, nil
}

func (worker *Worker) RandomState(t ShapeType, a int) *State {
	switch t {
	default:
		return worker.RandomState(ShapeType(worker.Rnd.Intn(8)+1), a)
	case ShapeTypeTriangle:
		return NewState(worker, NewRandomTriangle(worker), a)
	case ShapeTypeRectangle:
		return NewState(worker, NewRandomRectangle(worker), a)
	case ShapeTypeEllipse:
		return NewState(worker, NewRandomEllipse(worker), a)
	case ShapeTypeCircle:
		return NewState(worker, NewRandomCircle(worker), a)
	case ShapeTypeRotatedRectangle:
		return NewState(worker, NewRandomRotatedRectangle(worker), a)
	case ShapeTypeQuadratic:
		return NewState(worker, NewRandomQuadratic(worker), a)
	case ShapeTypeRotatedEllipse:
		return NewState(worker, NewRandomRotatedEllipse(worker), a)
	case ShapeTypePolygon:
		return NewState(worker, NewRandomPolygon(worker, 4, false), a)
	}
}
