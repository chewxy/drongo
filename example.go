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

	for _, neutral := range ns {
		dep, err := loadOne(neutral, Neutral)
		if err != nil {
			return err
		}
		ex := example{dep, Neutral}
		examples = append(examples, ex)
	}

	for _, lib := range ls {
		dep, err := loadOne(lib, Liberal)
		if err != nil {
			return err
		}
		ex := example{dep, Liberal}
		examples = append(examples, ex)
	}

	for _, con := range cs {
		dep, err := loadOne(con, Conservative)
		if err != nil {
			return err
		}
		ex := example{dep, Conservative}
		examples = append(examples, ex)
	}
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
