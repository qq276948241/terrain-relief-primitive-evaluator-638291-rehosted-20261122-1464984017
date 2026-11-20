package primitive

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"image/color/palette"
	"image/draw"
	"image/gif"
	"image/jpeg"
	"image/png"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
)

func LoadImage(path string) (image.Image, error) {
	if path == "-" {
		im, _, err := image.Decode(os.Stdin)
		return im, err
	} else {
		file, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		defer file.Close()
		im, _, err := image.Decode(file)
		return im, err
	}
}

// saveAtomic writes output to a temporary file in the destination directory
// and renames it into place only after the write and close succeed. The
// temporary file is removed on any failure, so an interrupted or failed
// write never leaves a partial file behind and never clobbers an existing
// output with a half-written one.
func saveAtomic(path string, write func(w io.Writer) error) (err error) {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".primitive-tmp-*")
	if err != nil {
		return fmt.Errorf("cannot create temporary file in %s: %w", dir, err)
	}
	tmpName := tmp.Name()
	defer func() {
		if err != nil {
			tmp.Close()
			os.Remove(tmpName)
		}
	}()
	if err = write(tmp); err != nil {
		return fmt.Errorf("cannot write %s: %w", path, err)
	}
	if err = tmp.Close(); err != nil {
		return fmt.Errorf("cannot write %s: %w", path, err)
	}
	if err = os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("cannot move output into place at %s: %w", path, err)
	}
	return nil
}

func SaveFile(path, contents string) error {
	if path == "-" {
		_, err := fmt.Fprint(os.Stdout, contents)
		return err
	}
	return saveAtomic(path, func(w io.Writer) error {
		_, err := io.WriteString(w, contents)
		return err
	})
}

func SavePNG(path string, im image.Image) error {
	return saveAtomic(path, func(w io.Writer) error {
		return png.Encode(w, im)
	})
}

func SaveJPG(path string, im image.Image, quality int) error {
	return saveAtomic(path, func(w io.Writer) error {
		return jpeg.Encode(w, im, &jpeg.Options{quality})
	})
}

func SaveGIF(path string, frames []image.Image, delay, lastDelay int) error {
	g := gif.GIF{}
	for i, src := range frames {
		dst := image.NewPaletted(src.Bounds(), palette.Plan9)
		draw.Draw(dst, dst.Rect, src, image.ZP, draw.Src)
		g.Image = append(g.Image, dst)
		if i == len(frames)-1 {
			g.Delay = append(g.Delay, lastDelay)
		} else {
			g.Delay = append(g.Delay, delay)
		}
	}
	return saveAtomic(path, func(w io.Writer) error {
		return gif.EncodeAll(w, &g)
	})
}

func SaveGIFImageMagick(ctx context.Context, path string, frames []image.Image, delay, lastDelay int) (err error) {
	dir, err := os.MkdirTemp("", "")
	if err != nil {
		return err
	}
	// Always clean up the temporary frames, including on cancel and error.
	defer os.RemoveAll(dir)
	for i, im := range frames {
		if err := ctx.Err(); err != nil {
			return err
		}
		path := filepath.Join(dir, fmt.Sprintf("%06d.png", i))
		if err := SavePNG(path, im); err != nil {
			return err
		}
	}
	args := []string{
		"-loop", "0",
		"-delay", fmt.Sprint(delay),
		filepath.Join(dir, "*.png"),
		"-delay", fmt.Sprint(lastDelay - delay),
		filepath.Join(dir, fmt.Sprintf("%06d.png", len(frames)-1)),
		path,
	}
	cmd := exec.CommandContext(ctx, "convert", args...)
	if err := cmd.Run(); err != nil {
		return err
	}
	return nil
}

func NumberString(x float64) string {
	suffixes := []string{"", "k", "M", "G"}
	for _, suffix := range suffixes {
		if x < 1000 {
			return fmt.Sprintf("%.1f%s", x, suffix)
		}
		x /= 1000
	}
	return fmt.Sprintf("%.1f%s", x, "T")
}

func radians(degrees float64) float64 {
	return degrees * math.Pi / 180
}

func degrees(radians float64) float64 {
	return radians * 180 / math.Pi
}

func clamp(x, lo, hi float64) float64 {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}

func clampInt(x, lo, hi int) int {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func rotate(x, y, theta float64) (rx, ry float64) {
	rx = x*math.Cos(theta) - y*math.Sin(theta)
	ry = x*math.Sin(theta) + y*math.Cos(theta)
	return
}

func imageToRGBA(src image.Image) *image.RGBA {
	dst := image.NewRGBA(src.Bounds())
	draw.Draw(dst, dst.Rect, src, image.ZP, draw.Src)
	return dst
}

func copyRGBA(src *image.RGBA) *image.RGBA {
	dst := image.NewRGBA(src.Bounds())
	copy(dst.Pix, src.Pix)
	return dst
}

func uniformRGBA(r image.Rectangle, c color.Color) *image.RGBA {
	im := image.NewRGBA(r)
	draw.Draw(im, im.Bounds(), &image.Uniform{c}, image.ZP, draw.Src)
	return im
}

func AverageImageColor(im image.Image) color.NRGBA {
	rgba := imageToRGBA(im)
	size := rgba.Bounds().Size()
	w, h := size.X, size.Y
	var r, g, b int
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := rgba.RGBAAt(x, y)
			r += int(c.R)
			g += int(c.G)
			b += int(c.B)
		}
	}
	r /= w * h
	g /= w * h
	b /= w * h
	return color.NRGBA{uint8(r), uint8(g), uint8(b), 255}
}
