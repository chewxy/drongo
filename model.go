package main

import (
	"io/ioutil"
	"strings"

	. "github.com/chewxy/gorgonia"
	"github.com/chewxy/gorgonia/tensor"
	"github.com/chewxy/lingo"
	"github.com/chewxy/lingo/corpus"
	"github.com/pkg/errors"
)

var hiddenSizes = []int{100, 30}

type Model struct {
	// dictionaries and the like
	c *corpus.Corpus

	// neural network
	g   *ExprGraph
	t   tensor.Dtype
	emb *Node   // (n, d) matrix. n = vocabulary size; d = dims
	l0  *Banana // (d, h0) matrices. First layer GRU
	l1  *Banana // (h0, h1) matrices. Second layer GRU
	a   *Attn   // (d, d) matrix. attention layer:
	p   *Node   // (cat, d) matrixweights for softmax

	// dummy
	prev0 *Node
	prev1 *Node
}

func NewModel(embShape tensor.Shape, t tensor.Dtype, q, cats int) *Model {
	d := embShape[1]

	g := NewGraph()
	emb := NewMatrix(g, t, WithShape(embShape...), WithName("WordEmbedding"))
	l0 := NewGRU("gru-0", g, d, hiddenSizes[0], t)
	l1 := NewGRU("gru-1", g, hiddenSizes[0], hiddenSizes[1], t)
	attn := NewAttn("attention", g, tensor.Shape{hiddenSizes[1], hiddenSizes[1]}, t)
	p := NewMatrix(g, t, WithShape(cats, hiddenSizes[1]), WithInit(GlorotU(1)), WithName("FinalLayer"))

	prev0 := NewVector(g, t, WithShape(hiddenSizes[0]), WithInit(Zeroes()), WithName("DummyPrev0"))
	prev1 := NewVector(g, t, WithShape(hiddenSizes[1]), WithInit(Zeroes()), WithName("DummyPrev1"))

	return &Model{
		g:   g,
		t:   t,
		emb: emb,
		l0:  l0,
		l1:  l1,
		a:   attn,
		p:   p,

		prev0: prev0,
		prev1: prev1,
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

func (m *Model) OneWord(wordID int, prev0, prev1 *Node) (h0, h1, e *Node, err error) {
	if prev0 == nil {
		prev0 = m.prev0
	}
	if prev1 == nil {
		prev1 = m.prev1
	}

	input := Must(Slice(m.emb, S(wordID)))
	if h0, err = m.l0.Activate(input, prev0); err != nil {
		return
	}

	var dropped *Node
	if dropped, err = Dropout(h0, 0.5); err != nil {
		return
	}

	if h1, err = m.l1.Activate(dropped, prev1); err != nil {
		return
	}

	if e, err = m.a.Exp(h1); err != nil {
		return
	}
	return
}

func (m *Model) Fwd(s lingo.AnnotatedSentence) (prob *Node, err error) {
	hiddens := make(Nodes, 0, len(s))
	exps := make(Nodes, 0, len(s))
	var runningSum *Node

	var prev0, prev1 *Node
	for i, a := range s[1:] {
		if i == 0 {
			prev0 = m.prev0
			prev1 = m.prev1
		}

		var h0, h1, e *Node
		if h0, h1, e, err = m.OneWord(m.WordID(a), prev0, prev1); err != nil {
			return
		}

		hiddens = append(hiddens, h1)
		exps = append(exps, e)
		if runningSum == nil {
			runningSum = e
		} else {
			if runningSum, err = m.a.Sum(runningSum, e); err != nil {
				return
			}
		}
		prev0 = h0
		prev1 = h1
	}

	// build context nodes
	var context *Node
	for i, h := range hiddens {
		var weight, ctx *Node
		if weight, err = HadamardDiv(exps[i], runningSum); err != nil {
			return
		}

		if ctx, err = HadamardProd(weight, h); err != nil {
			ioutil.WriteFile("error.dot", []byte(h.RestrictedToDot(2, 9)), 0644)
			return
		}

		if context == nil {
			context = ctx
			continue
		}
		if context, err = Add(context, ctx); err != nil {
			return
		}
	}

	var finalLayer *Node
	if finalLayer, err = Mul(m.p, context); err != nil {
		return
	}
	return SoftMax(finalLayer)
}

func (m *Model) CostFn(s lingo.AnnotatedSentence, target Target) (cost *Node, err error) {
	var prob *Node
	if prob, err = m.Fwd(s); err != nil {
		err = errors.Wrap(err, "FWD")
		return
	}
	logProb := Must(Neg(Must(Log(prob))))
	return Slice(logProb, S(int(target)))
}

func (m *Model) Train(solver Solver, pair example) (c float64, err error) {
	var g *ExprGraph
	var cost *Node
	if cost, err = m.CostFn(pair.dep.AnnotatedSentence, pair.target); err != nil {
		return
	}

	g = m.g.SubgraphRoots(cost)

	// f, _ := os.OpenFile("LOOOOG", os.O_APPEND|os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	// logger := log.New(f, "", 0)

	// machine := NewLispMachine(g, WithLogger(logger), WithWatchlist(), LogBothDir())
	machine := NewLispMachine(g)
	if err = machine.RunAll(); err != nil {
		if ctxerr, ok := err.(contextualError); ok {
			ioutil.WriteFile("error.dot", []byte(ctxerr.Node().RestrictedToDot(2, 9)), 0644)
		}
		return
	}

	v := cost.Value()
	switch v.Dtype() {
	case Float32:
		c = float64(v.Data().(float32))
	case Float64:
		c = v.Data().(float64)
	}

	// machine.UnbindAll()

	err = solver.Step(m.Learnables())
	return
}

func (m *Model) PredPreparsed(dep *lingo.Dependency) (class Target, err error) {
	var prob *Node
	if prob, err = m.Fwd(dep.AnnotatedSentence); err != nil {
		err = errors.Wrap(err, "Fwd failed")
		return
	}
	g := m.g.SubgraphRoots(prob)
	machine := NewLispMachine(g, ExecuteFwdOnly())
	if err = machine.RunAll(); err != nil {
		return
	}

	val := prob.Value().(tensor.Tensor)

	var t tensor.Tensor
	if t, err = tensor.Argmax(val, 0); err != nil {
		return
	}
	return Target(t.ScalarValue().(int)), nil

}

func (m *Model) Pred(s string) (class Target, err error) {
	var dep *lingo.Dependency
	if dep, err = pipeline(s, strings.NewReader(s)); err != nil {
		err = errors.Wrap(err, "Basic NLP pipeline failed")
		return
	}
	return m.PredPreparsed(dep)
}
