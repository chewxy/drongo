package main

import (
	"fmt"
	"io"
	"strings"

	"github.com/chewxy/lingo"
	"github.com/peterh/liner"
)

type ctx struct {
	*liner.State
	promptStr string

	q     string
	dep   *lingo.Dependency
	class Target
	m     *Model
}

func newCtx(m *Model) *ctx {
	l := liner.NewLiner()
	l.SetCtrlCAborts(true)
	return &ctx{
		State:     l,
		promptStr: ">>>",
		m:         m,
	}
}

func (c *ctx) Run() {
	for {
		err := c.main()
		if err == io.EOF {
			break
		}
	}
}

func (c *ctx) main() (err error) {
	var q string
	var dep *lingo.Dependency
	var class Target
	if q, err = c.Prompt(c.promptStr); err != nil {
		if err == io.EOF {
			return
		}
		goto end
	}

	if q == "" {
		return nil
	}

	c.AppendHistory(q)
	if strings.HasPrefix(q, ":") {
		// process commands
		switch q {
		case ":dep":
			if c.dep != nil {
				fmt.Printf("%v\n", c.dep.SprintRel())
			} else {
				fmt.Println("No Dependency yet")
			}
		case ":q":
			fmt.Printf("%q\n", c.q)
		}
		goto end
	}
	if q == c.q {
		fmt.Printf("Predicted; %s\n", c.class)
		goto end
	}

	if dep, err = pipeline(q, strings.NewReader(q)); err != nil {
		goto end
	}

	if class, err = c.m.PredPreparsed(dep); err != nil {
		goto end
	}

	// save state
	c.q = q
	c.dep = dep
	c.class = class
	fmt.Println("Predicted: %s\n", class)

end:
	// Catch errors except EOF
	if err != nil {
		fmt.Printf("ERR: %v\n", err)
		return nil
	}

	return nil
}
