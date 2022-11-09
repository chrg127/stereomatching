all: tesi.pdf

tesi.pdf: tesi.tex
	latexmk -f -pdf tesi.tex

.PHONY: clean

clean:
	-rm tesi.aux tesi.bbl tesi.blg tesi.fdb_latexmk tesi.log tesi.lof tesi.lot tesi.out tesi.pdf tesi.toc tesi.dvi tesi.fls tesi.synctex.gz
