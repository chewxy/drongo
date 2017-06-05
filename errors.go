package main

import (
	"fmt"

	"github.com/chewxy/gorgonia"
)

type nocomp string

func (e nocomp) Error() string     { return fmt.Sprintf("no %v", string(e)) }
func (e nocomp) Component() string { return string(e) }

type contextualError interface {
	Node() *gorgonia.Node
	Value() gorgonia.Value
	InstructionID() int
}
