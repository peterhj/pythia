# Pythia

Pythia is [Praline](https://github.com/peterhj/praline) v2 + [Meringue](https://github.com/peterhj/meringue) v2.

In its current form, Pythia is an interpreter for a language, also called Pythia, with Python-ish syntax.

The Pythia language is a _meta-language_ for a higher-order epistemic logic.

This repo is a snapshot of work-in-progress.
There are a small number (~50) of test cases, most of which are passing (though a handful spuriously).

## Motivation

Currently, our primary motivation is for Pythia to be an excellent programming environment in which to scalably implement a parser from natural language to logical forms (e.g. for auto-formalizing natural language mathematics).
Pythia can be understood as an operationalization of the program described in:

- https://peterhj.github.io/notes/loglang.html
- https://peterhj.github.io/notes/parse.html
- https://peterhj.github.io/notes/roadmap.html

The design of Pythia intentionally addresses limitations in the implementation of [Praline](https://github.com/peterhj/praline) v1,
and also unifies the logical frameworks in both [Praline](https://github.com/peterhj/praline) and [Meringue](https://github.com/peterhj/meringue).

## Related Work

- [Dodona](https://arxiv.org/abs/2012.11401) (Daniel Selsam, Jesse Michael Han, Leonardo de Moura, Patrice Godefroid)
- [Dusa](https://arxiv.org/abs/2405.19040) (Chris Martens, Robert J. Simmons, Michael Arntzenius)
- [Executable semantic parsing](https://arxiv.org/abs/1603.06677) (Percy Liang)
- Prolog (Alain Colmerauer, Philippe Roussel, Robert Kowalski)

## License

Apache 2.0 License
