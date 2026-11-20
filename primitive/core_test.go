package primitive

import (
	"image"
	"image/color"
	"math/rand"
	"testing"
)

// makeGradientTarget builds a deterministic opaque test image.
func makeGradientTarget(w, h int) *image.RGBA {
	im := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := im.PixOffset(x, y)
			im.Pix[i+0] = uint8((x*7 + y*3) % 256)
			im.Pix[i+1] = uint8((x*2 + y*11) % 256)
			im.Pix[i+2] = uint8((x*5 + y*5 + 40) % 256)
			im.Pix[i+3] = 255
		}
	}
	return im
}

// makeTransparentTarget builds a fully transparent target image.
func makeTransparentTarget(w, h int) *image.RGBA {
	return image.NewRGBA(image.Rect(0, 0, w, h))
}

// makeTranslucentTarget builds a target with varying alpha per pixel.
func makeTranslucentTarget(w, h int) *image.RGBA {
	im := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := im.PixOffset(x, y)
			im.Pix[i+0] = uint8((x*9 + y) % 256)
			im.Pix[i+1] = uint8((x + y*4) % 256)
			im.Pix[i+2] = uint8((x*3 + y*7) % 256)
			im.Pix[i+3] = uint8((x*13 + y*17) % 256)
		}
	}
	return im
}

// checkPartialMatchesFull draws lines onto before and verifies that the
// incremental total/score exactly matches a naive full-image recomputation.
func checkPartialMatchesFull(t *testing.T, target, before *image.RGBA, c Color, lines []Scanline) {
	t.Helper()
	size := target.Bounds().Size()
	w, h := size.X, size.Y

	after := copyRGBA(before)
	drawLines(after, c, lines)

	baseTotal := differenceFullTotal(target, before)
	gotTotal := differencePartialTotal(target, before, after, baseTotal, lines)
	wantTotal := differenceFullTotal(target, after)
	if gotTotal != wantTotal {
		t.Fatalf("incremental total mismatch: got %d, full recompute says %d", gotTotal, wantTotal)
	}

	gotScore := totalToScore(gotTotal, w, h)
	wantScore := differenceFull(target, after)
	if gotScore != wantScore {
		t.Fatalf("incremental score mismatch: got %v, full recompute says %v", gotScore, wantScore)
	}

	// The compatibility wrapper round-trips the score through a float64 and
	// truncates back to an integer total, so it can drift by a tiny,
	// well-understood amount (less than one unit of squared error). The
	// exactness guarantee lives in the integer-total path asserted above;
	// this bound only documents the legacy wrapper's round-trip error.
	beforeScore := totalToScore(baseTotal, w, h)
	wrapperScore := differencePartial(target, before, after, beforeScore, lines)
	diff := wrapperScore - wantScore
	if diff < 0 {
		diff = -diff
	}
	if diff > 1e-6 {
		t.Fatalf("differencePartial wrapper drifted too far: got %v, want %v", wrapperScore, wantScore)
	}
}

func TestPartialMatchesFull(t *testing.T) {
	const w, h = 32, 24

	t.Run("opaque gradient target", func(t *testing.T) {
		target := makeGradientTarget(w, h)
		before := uniformRGBA(image.Rect(0, 0, w, h), color.NRGBA{200, 100, 50, 255})
		lines := []Scanline{{Y: 3, X1: 2, X2: 20, Alpha: 0xffff}, {Y: 4, X1: 5, X2: 25, Alpha: 0xffff}}
		checkPartialMatchesFull(t, target, before, Color{10, 200, 30, 128}, lines)
	})

	t.Run("transparent source image", func(t *testing.T) {
		target := makeTransparentTarget(w, h)
		before := uniformRGBA(image.Rect(0, 0, w, h), color.NRGBA{0, 0, 0, 0})
		lines := []Scanline{{Y: 0, X1: 0, X2: w - 1, Alpha: 0xffff}, {Y: h - 1, X1: 0, X2: w - 1, Alpha: 0xffff}}
		checkPartialMatchesFull(t, target, before, Color{255, 255, 255, 200}, lines)
	})

	t.Run("translucent layers stack", func(t *testing.T) {
		target := makeTranslucentTarget(w, h)
		before := uniformRGBA(image.Rect(0, 0, w, h), color.NRGBA{90, 90, 90, 255})
		// first translucent layer
		lines1 := []Scanline{{Y: 8, X1: 4, X2: 27, Alpha: 0x8000}, {Y: 9, X1: 4, X2: 27, Alpha: 0x4000}}
		after1 := copyRGBA(before)
		drawLines(after1, Color{200, 30, 30, 128}, lines1)
		// second translucent layer on top of the first
		lines2 := []Scanline{{Y: 8, X1: 10, X2: 15, Alpha: 0xffff}, {Y: 10, X1: 0, X2: 31, Alpha: 0x2000}}
		checkPartialMatchesFull(t, target, after1, Color{30, 30, 220, 64}, lines2)
	})

	t.Run("shape clipped by canvas edge", func(t *testing.T) {
		target := makeGradientTarget(w, h)
		before := uniformRGBA(image.Rect(0, 0, w, h), color.NRGBA{10, 250, 10, 255})
		// raw lines extend past every edge; cropScanlines must clip them
		raw := []Scanline{
			{Y: -3, X1: -10, X2: 100, Alpha: 0xffff},
			{Y: 0, X1: -5, X2: 8, Alpha: 0xffff},
			{Y: h - 1, X1: 20, X2: w + 7, Alpha: 0xffff},
			{Y: h + 2, X1: 0, X2: 10, Alpha: 0xffff},
		}
		lines := cropScanlines(raw, w, h)
		checkPartialMatchesFull(t, target, before, Color{1, 2, 3, 255}, lines)
	})

	t.Run("degenerate polygon covers nothing", func(t *testing.T) {
		target := makeGradientTarget(w, h)
		before := uniformRGBA(image.Rect(0, 0, w, h), color.NRGBA{77, 88, 99, 255})
		worker := NewWorker(target)
		worker.Init(before, differenceFullTotal(target, before))
		// all points identical: a zero-area polygon
		p := &Polygon{Worker: worker, Order: 4, Convex: false,
			X: []float64{5, 5, 5, 5}, Y: []float64{5, 5, 5, 5}}
		lines := p.Rasterize()
		checkPartialMatchesFull(t, target, before, Color{0, 0, 0, 128}, lines)
	})

	t.Run("empty line set leaves score untouched", func(t *testing.T) {
		target := makeGradientTarget(w, h)
		before := uniformRGBA(image.Rect(0, 0, w, h), color.NRGBA{1, 2, 3, 255})
		checkPartialMatchesFull(t, target, before, Color{9, 9, 9, 255}, nil)
	})

	t.Run("float boundary values", func(t *testing.T) {
		// identical images: total is exactly zero
		target := makeGradientTarget(w, h)
		before := copyRGBA(target)
		if got := differenceFullTotal(target, before); got != 0 {
			t.Fatalf("identical images: got total %d, want 0", got)
		}
		if got := differenceFull(target, before); got != 0 {
			t.Fatalf("identical images: got score %v, want 0", got)
		}
		// alpha extremes on a single pixel
		lines := []Scanline{{Y: 0, X1: 0, X2: 0, Alpha: 0xffff}}
		checkPartialMatchesFull(t, target, before, Color{0, 0, 0, 1}, lines)
		checkPartialMatchesFull(t, target, before, Color{255, 255, 255, 255}, lines)
		// zero-coverage scanline alpha
		faint := []Scanline{{Y: 2, X1: 2, X2: 6, Alpha: 1}}
		checkPartialMatchesFull(t, target, before, Color{128, 128, 128, 128}, faint)
	})
}

// TestPartialMatchesFullRandomShapes fuzzes every shape type with a fixed
// seed and verifies the incremental score against a full recompute. This
// catches partial-difference implementations that silently drop pixels
// outside the shape's bounding lines.
func TestPartialMatchesFullRandomShapes(t *testing.T) {
	const w, h = 48, 36
	target := makeTranslucentTarget(w, h)
	rnd := rand.New(rand.NewSource(1234))
	worker := NewWorker(target)
	worker.SetRand(rand.New(rand.NewSource(99)))

	current := uniformRGBA(image.Rect(0, 0, w, h), color.NRGBA{120, 130, 140, 255})
	total := differenceFullTotal(target, current)

	types := []ShapeType{
		ShapeTypeTriangle, ShapeTypeRectangle, ShapeTypeEllipse,
		ShapeTypeCircle, ShapeTypeRotatedRectangle, ShapeTypeQuadratic,
		ShapeTypeRotatedEllipse, ShapeTypePolygon,
	}
	for i := 0; i < 200; i++ {
		st := types[rnd.Intn(len(types))]
		state := worker.RandomState(st, 1+rnd.Intn(255))
		worker.Init(current, total)
		lines := state.Shape.Rasterize()
		color := computeColor(target, current, lines, state.Alpha)

		before := copyRGBA(current)
		drawLines(current, color, lines)
		total = differencePartialTotal(target, before, current, total, lines)

		want := differenceFullTotal(target, current)
		if total != want {
			t.Fatalf("iteration %d (%v): incremental total %d != full total %d",
				i, st, total, want)
		}
	}
}
