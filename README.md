# texbuild

`texbuild` is a build system for LaTeX documents.
It provides namespacing for labeled items.
Consider a LaTeX document composed of a main file
```
% main.tex

\documentclass{article}
\begin{document}
This is an equation
\begin{equation}
  f(x) = x \label{equation}
\end{equation}
\input{foo.tex}
Let's refer to Equation~(\ref{equation}).
\end{document}
```
which includes a subordinate file
```
% foo.tex

\begin{equation}
  f(x) = x^2 \label{equation}
\end{equation}
```
In this situation, `\ref{equation}` is ambiguous because both files use `\label{equation}` and the namespace for labeled items in LaTeX is global.
`texbuild` fixes this by supporting import semantics:
```
% main.tex

% -- begin imports --
% import foo as foo
% -- end imports --

\documentclass{article}
\begin{document}
This is an equation
\begin{equation}
  f(x) = x \label{equation} % export
\end{equation}
\input{foo.tex}
We can refer to foo's Equation~(\ref{foo.equation}) or our own Equation~(\ref{equation}).
\end{document}
```
and
```
% foo.tex

\begin{equation}
  f(x) = x^2 \label{equation} % export
\end{equation}
```

## Example

To see `texbuild` in action, install it and use it to build the example LaTeX documemt provided in this repo, by following these steps:
```
$ pip install texbuild
$ git clone https://github.com/DanielSank/texbuild
$ cd texbuild/example
$ texbuild

This produces `texbuild/example/texbuild-out/main.pdf`, which illustrates the use of `texbuild`.
