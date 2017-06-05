package main

import (
	"flag"
	"log"
	"math/rand"
	"os"
	"runtime/pprof"

	"github.com/chewxy/gorgonia"
	"github.com/chewxy/gorgonia/tensor"
)

const (
	MAXQUERY = 45 // query length of 45 words is max. You can't analyze idiots like Nabokov or James Joyce but those are shitty writers anyway
)

const (
	lib001 = `As usual, then, government intervention into the market caused unintended, undesired consequences, but politicians blame the HMOs instead of the interventions that helped create them.`
	con001 = `They have transferred control of elections to the government bureaucracy they fund and control, created complex ballot-access laws, switched to the Australian ballot to weaken local parties, outlawed corporate contributions, and imposed contribution limits to make it hard for opponents to fund a credible challenge.`
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
	shuffleExamples(examples)
	shuffleExamples(examples)
	shuffleExamples(examples)

	trainingLen := int(0.85 * float64(len(examples)))
	trainingSet := make([]example, trainingLen)
	validateSet := make([]example, len(examples)-trainingLen)
	copy(trainingSet, examples)
	copy(validateSet, examples[trainingLen:])

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	emb := depModel.WordEmbeddings()
	m := NewModel(emb.Shape(), tensor.Float64, MAXQUERY, int(MAXTARGETS))
	m.c = depModel.Corpus()
	m.SetEmbed(emb)
	solver := gorgonia.NewAdaGradSolver(gorgonia.WithClip(3.0), gorgonia.WithL2Reg(0.000001))
	for i := 0; i < 20; i++ {
		if err := Train(i, m, solver, trainingSet); err != nil {
			log.Fatalf("Error while training during iteration %d: %+v", i, err)
		}
		shuffleExamples(trainingSet)

		if i%10 == 0 {
			acc, err := checkAcc(m, validateSet)
			if err != nil {
				log.Fatal(err)
			}
			log.Printf("Epoch %d. Accuracy: %f | %v \n", i, acc, len(validateSet))
		}
	}
	if *memprofile != "" {
		f, err := os.Create(*memprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.WriteHeapProfile(f)
		f.Close()
	}

	c := newCtx(m)
	defer c.Close()
	c.Run()

}

func Train(epoch int, m *Model, solver gorgonia.Solver, trainingSet []example) (err error) {
	costs := make([]float64, len(trainingSet))
	for i, ex := range trainingSet {
		var cost float64
		if cost, err = m.Train(solver, ex); err != nil {
			return
		}
		costs[i] = cost
	}
	log.Printf("Epoch %d. Avg Cost %f", epoch, averageCosts(costs))
	return nil
}

func checkAcc(m *Model, validationset []example) (acc float64, err error) {
	var correct float64
	for _, ex := range validationset {
		var class Target
		if class, err = m.PredPreparsed(ex.dep); err != nil {
			return
		}
		if class == ex.target {
			correct++
		}
	}
	return correct / float64(len(validationset)), nil
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
