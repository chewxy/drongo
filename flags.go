package main

import "flag"

var (
	posModelLoc = flag.String("pos", "", "Location for the POSTagger Model")
	depModelLoc = flag.String("dep", "", "Location for the Dependency Parsing Model")
	clusterLoc  = flag.String("cluster", "", "Location for brown cluster text file")
	cpuprofile  = flag.String("cpuprofile", "", "CPU Profile Location")
	memprofile  = flag.String("memprofile", "", "Mem Profile Location")
)
