package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"

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
	copy(trainingSet, examples)

	emb := depModel.WordEmbeddings()
	m := NewModel(emb.Shape(), tensor.Float64, MAXQUERY, int(MAXTARGETS))
	m.c = depModel.Corpus()
	m.SetEmbed(emb)
	solver := gorgonia.NewAdamSolver(gorgonia.WithClip(3.0), gorgonia.WithLearnRate(0.0005), gorgonia.WithL2Reg(0.000001))
	for i := 0; i < 300; i++ {
		if err := Train(i, m, solver, trainingSet); err != nil {
			log.Fatalf("Error while training during iteration %d: %+v", i, err)
		}
		shuffleExamples(trainingSet)
	}

	t0, err := m.Pred(lib001)
	if err != nil {
		log.Fatal(err)
	}

	t1, err := m.Pred(con001)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("T0: %v | T1: %v\n", t0, t1)
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
