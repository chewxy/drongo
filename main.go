package main

import (
	"flag"
	"log"

	"github.com/chewxy/gorgonia"
	"github.com/chewxy/gorgonia/tensor"
)

const (
	MAXQUERY = 45 // query length of 45 words is max. You can't analyze idiots like Nabokov or James Joyce but those are shitty writers anyway
)

func main() {
	flag.Parse()
	if err := loadModels(); err != nil {
		log.Fatal(err)
	}
	if err := loadExamples(); err != nil {
		log.Fatal(err)
	}

	emb := depModel.WordEmbeddings()
	m := NewModel(emb.Shape(), tensor.Float64, MAXQUERY, int(MAXTARGETS))
	m.c = depModel.Corpus()
	m.SetEmbed(emb)
	solver := gorgonia.NewAdamSolver()
	for i := 0; i < 100; i++ {
		if err := m.Train(i, solver); err != nil {
			log.Fatalf("Error while training during iteration %d: %+v", i, err)
		}
	}
}
