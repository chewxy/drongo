package main

import "fmt"

type nocomp string

func (e nocomp) Error() string     { return fmt.Sprintf("no %v", string(e)) }
func (e nocomp) Component() string { return string(e) }
