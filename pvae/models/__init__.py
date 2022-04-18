from .tabular import Tree as VAE_tree, CSV as VAE_csv, SimTreeDistortion as AE_simtree
from .mnist import Mnist as VAE_mnist

__all__ = [VAE_csv, VAE_tree, AE_simtree, VAE_mnist]