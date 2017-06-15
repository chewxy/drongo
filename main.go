package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"runtime/pprof"

	"github.com/chewxy/gorgonia"
	"github.com/chewxy/gorgonia/tensor"
	"github.com/pkg/profile"
)

const (
	MAXQUERY  = 45 // query length of 45 words is max. You can't analyze idiots like Nabokov or James Joyce but those are shitty writers anyway
	partition = 0.85
)

func main() {
	flag.Parse()
	rand.Seed(1337)
	if err := loadModels(); err != nil {
		log.Fatal(err)
	}
	if err := loadExamples(); err != nil {
		log.Fatal(err)
	}
	log.Printf("Everything loaded. Start training. %d examples. %d validations", len(examples), len(validates))
	shuffleExamples(examples)

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	// write and then close the file. This file will be re-written
	// if *memprofile != "" {
	// 	f, err := os.Create(*memprofile)
	// 	if err != nil {
	// 		log.Fatal(err)
	// 	}
	// 	pprof.WriteHeapProfile(f)
	// 	f.Close()
	// 	runtime.GC()
	// }
	if *memprofile != "" {
		defer profile.Start(profile.MemProfile, profile.ProfilePath(".")).Stop()
	}

	emb := depModel.WordEmbeddings()
	m := NewModel(emb.Shape(), Float, MAXQUERY, int(MAXTARGETS))
	m.c = depModel.Corpus()
	m.SetEmbed(emb)
	solver := gorgonia.NewAdaGradSolver(gorgonia.WithClip(3.0), gorgonia.WithLearnRate(0.05), gorgonia.WithL2Reg(0.000001))
	for i := 0; i < 5; i++ {
		var cost float64
		var err error
		if cost, err = Train(i, m, solver, examples); err != nil {
			log.Fatalf("Error while training during iteration %d: %+v", i, err)
		}

		acc, f1, con, err := checkAcc(m, validates)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("%d | %f | %f | %f\n", i, cost, acc, f1)

		if i%10 == 0 || i < 10 {
			fmt.Printf("%+v\n", con)
		}
		shuffleExamples(examples)
	}
	// if *memprofile != "" {
	// 	f, err := os.Create(*memprofile)
	// 	if err != nil {
	// 		log.Fatal(err)
	// 	}
	// 	pprof.WriteHeapProfile(f)
	// 	f.Close()
	// }

	// c := newCtx(m)
	// defer c.Close()
	// c.Run()

}

func Train(epoch int, m *Model, solver gorgonia.Solver, trainingSet []example) (avgCost float64, err error) {
	costs := make([]float64, len(trainingSet))
	for i, ex := range trainingSet {
		var cost float64
		if cost, err = m.Train(solver, ex); err != nil {
			return
		}
		costs[i] = cost
	}
	return averageCosts(costs), nil
}

func checkAcc(m *Model, validationset []example) (acc, f1 float64, confusion tensor.Tensor, err error) {
	// row == pred, col == actual
	confusion = tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(int(MAXTARGETS), int(MAXTARGETS)))

	var correct float64
	for _, ex := range validationset {
		var class Target
		if class, err = m.PredPreparsed(ex.dep); err != nil {
			return
		}
		if class == ex.target {
			correct++
		}

		var s tensor.Tensor
		if s, err = confusion.Slice(gorgonia.S(int(class))); err != nil {
			return
		}
		s.Data().([]float64)[int(ex.target)] += 1
	}

	var sumF1s float64
	sumClasses0, _ := tensor.Sum(confusion, 0)
	sumClasses1, _ := tensor.Sum(confusion, 1)
	for i := Neutral; i < MAXTARGETS; i++ {
		truePosI, _ := confusion.At(int(i), int(i))
		sum1I, _ := sumClasses1.At(int(i))

		truePos := truePosI.(float64)
		sum1 := sum1I.(float64)
		prec := truePos / (sum1 + 1e-4)

		sum0I, _ := sumClasses0.At(int(i))
		sum0 := sum0I.(float64)
		recall := truePos / (sum0 + 1e-4)

		f1 := 2 * (prec * recall) / (prec + recall - 1e-8)
		sumF1s += f1
	}
	f1 = sumF1s / float64(MAXTARGETS)
	acc = correct / float64(len(validationset))
	return
}

func shuffleExamples(a []example) {
	for i := range a {
		j := rand.Intn(i + 1)
		a[i], a[j] = a[j], a[i]
	}
}

func averageCosts(a []float64) (retVal float64) {
	for _, v := range a {
		retVal += v
	}
	return retVal / float64(len(a))
}
