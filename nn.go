package main

import (
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
	g  *ExprGraph
	wz *Node
	wr *Node
	w  *Node

	Name   string // optional name
	Prev   *Node  // previous hidden, which is an additional input
	Hidden *Node  // output
}

func NewGRU(name string, g *ExprGraph, input tensor.Shape, t tensor.Dtype) *Banana {
	wz := NewMatrix(g, t, WithShape(input...), WithName(fmt.Sprintf("%v.wz", name)), WithInit(GlorotN(1.0)))
	wr := NewMatrix(g, t, WithShape(input...), WithName(fmt.Sprintf("%v.wr", name)), WithInit(GlorotN(1.0)))
	w := NewMatrix(g, t, WithShape(input...), WithName(fmt.Sprintf("%v.w", name)), WithInit(GlorotN(1.0)))

	gru := &Banana{
		g:  g,
		wz: wz,
		wr: wr,
		w:  w,
	}
	return gru
}

func (l *Banana) Activate(x *Node) (retVal *Node, err error) {
	if l.Prev == nil {
		return nil, errors.Errorf("Expected a previous state")
	}
	var cat *Node
	if cat, err = Concat(0, l.Prev, x); err != nil {
		return
	}

	// update gate
	var wzCat, zt *Node
	if wzCat, err = Mul(l.wz, cat); err != nil {
		return
	}
	if zt, err = Sigmoid(wzCat); err != nil {
		return
	}

	// reset gate
	var wrCat, rt *Node
	if wrCat, err = Mul(l.wr, cat); err != nil {
		return
	}
	if rt, err = Sigmoid(wrCat); err != nil {
		return
	}

	// hidden gate h~
	var reset, resetCat, wResetCat, h2 *Node
	if reset, err = HadamardProd(rt, l.Prev); err != nil {
		return
	}

	if resetCat, err = Concat(0, reset, x); err != nil {
		return
	}

	if wResetCat, err = Mul(l.w, resetCat); err != nil {
		return
	}

	if h2, err = Tanh(wResetCat); err != nil {
		return
	}

	dt := l.w.Value().Dtype()
	oneV := tensor.Ones(dt, zt.Shape()...)
	one := l.g.Constant(oneV)

	var onemzt, gatePrev, gateh2 *Node
	if onemzt, err = Sub(one, zt); err != nil {
		return
	}
	if gatePrev, err = HadamardProd(onemzt, l.Prev); err != nil {
		return
	}
	if gateh2, err = HadamardProd(zt, h2); err != nil {
		return
	}

	l.Hidden, err = Add(gatePrev, gateh2)
	return l.Hidden, err
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
		w:  NewMatrix(g, t, WithShape(shape...), WithInit(GlorotN(1))),
	}
}

func (l *Attn) Exp(x *Node) (retVal *Node, err error) {
	var wx, e *Node
	if wx, err = Mul(l.w, x); err != nil {
		return
	}
	if e, err = l.Fn(wx); err != nil {
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
