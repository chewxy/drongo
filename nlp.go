package main

import (
	"io"
	"os"

	"github.com/chewxy/lingo"
	"github.com/chewxy/lingo/dep"
	"github.com/chewxy/lingo/lexer"
	"github.com/chewxy/lingo/pos"
	"github.com/chewxy/wordnet"
	"github.com/kljensen/snowball"
)

var (
	posModel *pos.Model
	depModel *dep.Model
	clusters map[string]lingo.Cluster
)

const (
	defaultPOSModelLoc = "model/shared/pos_stanfordtags_universalrel.final.model"
	defaultDepModelLoc = "model/shared/dep_stanfordtags_universalrel.final.model"
	defaultClusterLoc  = "model/shared/clusters.txt"
)

func init() {
	wordnet.Init("model/shared/wordnet")
}

type stemmer struct{}

func (stemmer) Stem(a string) (string, error) {
	return snowball.Stem(a, "english", true)
}

type fixer struct {
	stemmer
}

func (fixer) Clusters() (map[string]lingo.Cluster, error) { return clusters, nil }
func (fixer) Lemmatize(a string, tag lingo.POSTag) ([]string, error) {
	return wordnet.Lemmatize(a, tag), nil
	// return nil, nocomp("No Lemmatizer")
}

func loadModels() (err error) {
	pl := defaultPOSModelLoc
	dl := defaultDepModelLoc
	cl := defaultClusterLoc

	if *posModelLoc != "" {
		pl = *posModelLoc
	}
	if *depModelLoc != "" {
		dl = *depModelLoc
	}
	if *clusterLoc != "" {
		cl = *clusterLoc
	}

	if posModel, err = pos.Load(pl); err != nil {
		return
	}
	if depModel, err = dep.Load(dl); err != nil {
		return
	}

	var f io.ReadCloser
	if f, err = os.Open(cl); err != nil {
		switch {
		case os.IsNotExist(err):
			return nil
		case os.IsPermission(err):
			return nil
		default:
			return err
		}
	}
	clusters = lingo.ReadCluster(f)
	f.Close()

	return nil
}

func pipeline(name string, f io.Reader) (*lingo.Dependency, error) {
	consOpts := []pos.ConsOpt{
		pos.WithModel(posModel),
		pos.WithStemmer(stemmer{}),
		pos.WithLemmatizer(fixer{}),
	}
	if clusters != nil {
		consOpts = append(consOpts, pos.WithCluster(clusters))
	}

	l := lexer.New(name, f)
	p := pos.New(consOpts...)
	d := dep.New(depModel)

	// set up pipeline
	p.Input = l.Output
	d.Input = p.Output
	go l.Run()
	go p.Run()
	go d.Run()

	select {
	case err := <-l.Errors:
		return nil, err
	case err := <-d.Error:
		return nil, err
	case dep := <-d.Output:
		return dep, nil
	}
	panic("Unreachable")
}
