package main

import (
	"log"
	"math/rand"

	. "github.com/chewxy/gorgonia"
	"github.com/chewxy/gorgonia/tensor"
	"github.com/chewxy/lingo"
	"github.com/chewxy/lingo/corpus"
)

type Model struct {
	// dictionaries and the like
	c *corpus.Corpus

	// neural network
	g     *ExprGraph
	t     tensor.Dtype
	emb   *Node   // (n, d) matrix. n = vocabulary size; d = dims
	input *Node   // (d) vector. vector sliced from emb
	l0    *Banana // (d, 2d) matrices. one layer GRU; 2d = 2x d
	a     *Attn   // (d, |q|) matrix. attention layer: |q| = max sentence length
	p     *Node   // (cat, d) matrixweights for softmax

	// dummy
	prev *Node
}

func NewModel(embShape tensor.Shape, t tensor.Dtype, q, cats int) *Model {
	d := embShape[1]

	g := NewGraph()
	emb := NewMatrix(g, t, WithShape(embShape...))
	in := Must(Slice(emb, S(0))) // dummy slice
	l0 := NewGRU("gru-0", g, tensor.Shape{d, 2 * d}, t)
	attn := NewAttn("attention", g, tensor.Shape{d, q}, t)
	p := NewMatrix(g, t, WithShape(cats, d), WithInit(GlorotN(1)))

	prev := NewVector(g, t, WithShape(d), WithInit(Zeroes()))

	return &Model{
		g:     g,
		t:     t,
		emb:   emb,
		input: in,
		l0:    l0,
		a:     attn,
		p:     p,

		prev: prev,
	}
}

func (m *Model) SetEmbed(emb Value) {
	Let(m.emb, emb)
}

func (m *Model) Learnables() Nodes {
	return Nodes{
		m.emb, m.l0.w, m.l0.wr, m.l0.wz, m.a.w, m.p, // todo: fix to use getters
	}
}

func (m *Model) WordID(a *lingo.Annotation) int {
	if id, ok := m.c.Id(a.Value); ok {
		return id
	}
	id, _ := m.c.Id("-UNKNOWN-")
	return id
}

func (m *Model) Fwd(wordID int, prev *Node) (h, e *Node, err error) {
	if err = UnsafeLet(m.input, S(wordID)); err != nil {
		return
	}

	m.l0.Prev = prev
	if h, err = m.l0.Activate(m.input); err != nil {
		return
	}

	if e, err = m.a.Exp(h); err != nil {
		return
	}
	return
}

func (m *Model) CostFn(s lingo.AnnotatedSentence, target Target) (cost *Node, err error) {
	hiddens := make(Nodes, 0, len(s))
	exps := make(Nodes, 0, len(s))
	var runningSum *Node

	var prev *Node
	for i, a := range s[1:] {
		if i == 0 {
			prev = m.prev
		}

		var h, e *Node
		if h, e, err = m.Fwd(m.WordID(a), prev); err != nil {
			return
		}

		hiddens = append(hiddens, h)
		exps = append(exps, e)
		if runningSum == nil {
			runningSum = e
		} else {
			if runningSum, err = m.a.Sum(runningSum, e); err != nil {
				return
			}
		}
		prev = h
	}

	// build context nodes
	var context *Node
	for i, h := range hiddens {
		var weight, ctx *Node
		if weight, err = HadamardDiv(exps[i], runningSum); err != nil {
			return
		}

		if ctx, err = Mul(weight, h); err != nil {
			return
		}
		log.Printf("ctx %v", ctx.Shape())

		if context == nil {
			context = ctx
			continue
		}
		if context, err = Add(context, ctx); err != nil {
			return
		}
	}
	log.Printf("context %v", context.Shape())

	var prob *Node
	if prob, err = Mul(context, m.p); err != nil {
		return
	}

	logProb := Must(Neg(Must(Log(prob))))
	cost = Must(Slice(logProb, S(int(target))))
	return
}

func (m *Model) Train(iter int, solver Solver) (err error) {
	log.Printf("iter %d | %t", iter, m == nil)
	i := rand.Intn(len(examples))
	pair := examples[i]

	var g *ExprGraph
	var cost *Node
	if cost, err = m.CostFn(pair.dep.AnnotatedSentence, pair.target); err != nil {
		return
	}

	g = m.g.SubgraphRoots(cost)
	machine := NewLispMachine(g)
	if err = machine.RunAll(); err != nil {
		return
	}
	machine.UnbindAll()

	return solver.Step(m.Learnables())
}
