package main

import (
	"bytes"
	"encoding/gob"
	"fmt"

	. "github.com/chewxy/gorgonia"
	"github.com/chewxy/gorgonia/tensor"
	"github.com/pkg/errors"
)

type FC struct {
	g *ExprGraph
	w *Node
	b *Node

	Fn  func(*Node) (*Node, error)
	Act *Node // output
}

func NewFC(g *ExprGraph, input tensor.Shape, t tensor.Dtype, winit, binit InitWFn) *FC {
	if input.TotalSize() != 2 {
		panic("Expects a matrix shape")
	}

	w := NewMatrix(g, t, WithShape(input...), WithInit(winit))
	b := NewVector(g, t, WithShape(input[1]), WithInit(binit))
	l := &FC{
		g: g,
		w: w,
		b: b,

		Fn: Sigmoid,
	}
	return l
}

func (l *FC) Activate(x *Node) (retVal *Node, err error) {
	if l.Act != nil {
		return l.Act, nil
	}

	var xw, xwb *Node
	if xw, err = Mul(x, l.w); err != nil {
		return
	}

	if xwb, err = Add(xw, l.b); err != nil {
		return
	}

	l.Act, err = l.Fn(xwb)
	return l.Act, err
}

// Banana is a standard GRU node. Geddit?
type Banana struct {
	g *ExprGraph

	// weights for mem
	u *Node
	w *Node
	b *Node

	// update gate
	uz *Node
	wz *Node
	bz *Node

	// reset gate
	ur  *Node
	wr  *Node
	br  *Node
	one *Node

	Name string // optional name
}

func NewGRU(name string, g *ExprGraph, inputSize, hiddenSize int, dt tensor.Dtype) *Banana {
	// standard weights
	u := NewMatrix(g, dt, WithShape(hiddenSize, hiddenSize), WithName(fmt.Sprintf("%v.u", name)), WithInit(Gaussian(0, 0.08)))
	w := NewMatrix(g, dt, WithShape(hiddenSize, inputSize), WithName(fmt.Sprintf("%v.w", name)), WithInit(Gaussian(0, 0.08)))
	b := NewVector(g, dt, WithShape(hiddenSize), WithName(fmt.Sprintf("%v.b", name)), WithInit(Zeroes()))

	// update gate
	uz := NewMatrix(g, dt, WithShape(hiddenSize, hiddenSize), WithName(fmt.Sprintf("%v.uz", name)), WithInit(Gaussian(0, 0.08)))
	wz := NewMatrix(g, dt, WithShape(hiddenSize, inputSize), WithName(fmt.Sprintf("%v.wz", name)), WithInit(Gaussian(0, 0.08)))
	bz := NewVector(g, dt, WithShape(hiddenSize), WithName(fmt.Sprintf("%v.b_uz", name)), WithInit(Zeroes()))

	// reset gate
	ur := NewMatrix(g, dt, WithShape(hiddenSize, hiddenSize), WithName(fmt.Sprintf("%v.ur", name)), WithInit(Gaussian(0, 0.08)))
	wr := NewMatrix(g, dt, WithShape(hiddenSize, inputSize), WithName(fmt.Sprintf("%v.wr", name)), WithInit(Gaussian(0, 0.08)))
	br := NewVector(g, dt, WithShape(hiddenSize), WithName(fmt.Sprintf("%v.bz", name)), WithInit(Zeroes()))

	ones := tensor.Ones(dt, hiddenSize)
	one := g.Constant(ones)
	gru := &Banana{
		g: g,

		u: u,
		w: w,
		b: b,

		uz: uz,
		wz: wz,
		bz: bz,

		ur: ur,
		wr: wr,
		br: br,

		one: one,
	}
	return gru
}

func (l *Banana) Activate(x, prev *Node) (retVal *Node, err error) {
	// update gate
	// z := Must(Sigmoid(Must(Add(Must(Add(Must(Mul(l.uz, prev)), Must(l.wz, x))), l.bz))))
	uzh := Must(Mul(l.uz, prev))
	wzx := Must(Mul(l.wz, x))
	z := Must(Sigmoid(
		Must(Add(
			Must(Add(uzh, wzx)),
			l.bz))))

	// reset gate
	// r := Must(Sigmoid(Must(Add(Must(Add(Must(Mul(l.wr, x)), Must(Mul(l.ur, prev)), l.br))))))
	urh := Must(Mul(l.ur, prev))
	wrx := Must(Mul(l.wr, x))
	r := Must(Sigmoid(
		Must(Add(
			Must(Add(urh, wrx)),
			l.br))))

	// memory for hidden
	hiddenFilter := Must(Mul(l.u, Must(HadamardProd(r, prev))))
	wx := Must(Mul(l.w, x))
	mem := Must(Tanh(
		Must(Add(
			Must(Add(hiddenFilter, wx)),
			l.b))))

	omz := Must(Sub(l.one, z))
	omzh := Must(HadamardProd(omz, prev))
	upd := Must(HadamardProd(z, mem))
	retVal = Must(Add(upd, omzh))
	return
}

type Attn struct {
	g    *ExprGraph
	w    *Node
	Fn   func(*Node) (*Node, error)
	name string
}

func NewAttn(name string, g *ExprGraph, shape tensor.Shape, t tensor.Dtype) *Attn {
	return &Attn{
		g:  g,
		Fn: Tanh,
		w:  NewMatrix(g, t, WithShape(shape...), WithInit(GlorotN(1)), WithName(fmt.Sprintf("%s.w", name))),
	}
}

func (l *Attn) Exp(x *Node) (retVal *Node, err error) {
	// var wx, do, e *Node
	var wx, e *Node
	if wx, err = Mul(l.w, x); err != nil {
		err = errors.Wrap(err, "wx")
		return
	}

	// if do, err = Dropout(wx, 0.3); err != nil {
	// 	return
	// }
	if e, err = l.Fn(wx); err != nil {
		// if e, err = l.Fn(do); err != nil {
		return
	}
	return Exp(e)
}

func (l *Attn) Sum(a, b *Node) (retVal *Node, err error) {
	return Add(a, b)
}

func (l *Attn) Weight(a, sum *Node) (retVal *Node, err error) {
	return Div(a, sum)
}

func (l *Attn) GobEncode() (p []byte, err error) {
	var buf bytes.Buffer
	encoder := gob.NewEncoder(&buf)
	if err = encoder.Encode(l.w.Value()); err != nil {
		return
	}
	return buf.Bytes(), nil
}
