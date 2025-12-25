#! /usr/bin/env python3

import numpy as np
from tqdm import tqdm

from ramses.util import *
from ramses.prm import * 
from ramses.mar import * 
from tqdm import tqdm
from ramses.mod import *
from ramses.euclidio import *
from ramses.gausiano import Gauss

def entrena(dirPrm, dirMar, lisUni, ficMod, *ficGui, ClsMod=Gauss):
    """
    Entrena el modelo acústico
    """
    unidades = leeLis(lisUni)

    # Inicializamos el modelo 
    modelo = ClsMod(lisMod=lisUni)

    # Inicializamos el entrenamiento 
    modelo.inicMod()

    # Bucle para todas las señales de entrenamiento 
    for señal in tqdm(leeLis(*ficGui), ascii="·|-#"): 
        # leemos la señal y el contenido del fichero de marcas
        pathPrm = pathName(dirPrm, señal, 'prm')
        prm = leePrm(pathPrm)
        pathMar = pathName(dirMar, señal, 'mar')
        unidad =cogeTrn(pathMar)

        #Actualizamos la información del entrenamiento 
        modelo += prm, unidad

    # Recalculamos el modelo 
    modelo.calcMod()

    # Escribimos el modelo representante
    modelo.escMod(ficMod)    

if __name__ == "__main__":
    from docopt import docopt
    import sys

    usage=f"""
        Entrena un modelo acústico para el reconocimiento de las vocales
        usage:
            {sys.argv[0]} [options] <guia> ...
            {sys.argv[0]} -h | --help
            {sys.argv[0]} --version

        options:
            -p, --dirPrm PATH  directori de la señal parametrizada [default: .]
            -m, --dirMar PATH  Directorio con el contenido fonético de las señales [defalut: .]
            -l, --lisUni PATH  Fichero con la lista de unidades fonéticas [default: Lis/vocales.lis]
            -M, --ficMod PATH  Fichero con el modelo resultante [default: Mod/vocales.mod]
            -e, --execPrev SCRIPT   scripts de ejecución previa
            -C, --classMod CLASS  Clase que implementa el modelado acústico
        """
    args= docopt(usage, version="tecparla2025")
    dirPrm = args["--dirPrm"]
    dirMar = args["--dirMar"]
    lisUni = args["--lisUni"]
    ficMod = args["--ficMod"]
    if args["--execPrev"]: exec(open(args["--execPrev"]).read())
    ficGui = args["<guia>"]
    ClsMod = eval(args["--classMod"]) if args["--classMod"] else Modelo

    entrena(dirPrm, dirMar, lisUni, ficMod, *ficGui, ClsMod=ClsMod)


