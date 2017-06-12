package main

import (
	"log"
	"os"
	"path/filepath"
	"sync"

	"github.com/chewxy/lingo"
)

type Target int

const (
	Neutral Target = iota
	Liberal
	Conservative
	MAXTARGETS
)

func (t Target) String() string {
	switch t {
	case Neutral:
		return "Neutral"
	case Liberal:
		return "Liberal"
	case Conservative:
		return "Conservative"
	default:
		return "UNKNOWN"
	}
}

type example struct {
	dep    *lingo.Dependency
	target Target
}

var examples []example
var validates []example

func loadExamples() (err error) {
	var ns, ls, cs []string
	if ns, err = filepath.Glob("model/neutral/*.txt"); err != nil {
		return
	}

	if ls, err = filepath.Glob("model/liberal/*.txt"); err != nil {
		return
	}

	if cs, err = filepath.Glob("model/conservative/*.txt"); err != nil {
		return
	}

	neutrals := make([]example, len(ns))
	for i, neutral := range ns {
		dep, err := loadOne(neutral, Neutral)
		if err != nil {
			return err
		}
		ex := example{dep, Neutral}

		neutrals[i] = ex
	}

	libs := make([]example, len(ls))
	for i, lib := range ls {
		dep, err := loadOne(lib, Liberal)
		if err != nil {
			return err
		}
		ex := example{dep, Liberal}
		libs[i] = ex
	}

	cons := make([]example, len(cs))
	for i, con := range cs {
		dep, err := loadOne(con, Conservative)
		if err != nil {
			return err
		}
		ex := example{dep, Conservative}
		cons[i] = ex
	}

	// build up examples
	l := int(partition * float64(len(neutrals)))
	examples = append(examples, neutrals[:l]...)
	validates = append(validates, neutrals[l:]...)

	l = int(partition * float64(len(libs)))
	examples = append(examples, libs[:l]...)
	validates = append(validates, libs[l:]...)

	l = int(partition * float64(len(cons)))
	examples = append(examples, cons[:l]...)
	validates = append(validates, cons[l:]...)

	return nil
}

func loadOneMultithread(name string, t Target, exChan chan example, errChan chan error, wg *sync.WaitGroup) {
	defer wg.Done()
	f, err := os.Open(name)
	if err != nil {
		errChan <- err
		return
	}
	defer f.Close()

	var dep *lingo.Dependency
	if dep, err = pipeline(name, f); err != nil {
		errChan <- err
		return
	}
	log.Printf("name: %q | %v\n", dep.ValueString(), dep.SprintRel())
	ex := example{dep, t}
	exChan <- ex
}

func loadOne(name string, t Target) (dep *lingo.Dependency, err error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return pipeline(name, f)
}
