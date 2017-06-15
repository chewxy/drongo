// +build !f64

package main

import "github.com/chewxy/gorgonia/tensor"

var Float = tensor.Float32

const (
	defaultDepModelLoc = "model/shared/dep_stanfordtags_universalrel.final.model_f32"
)
