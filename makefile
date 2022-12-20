.PHONY: help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	@python setup.py install

install_silent: ## Install dependencies (silent)
	@python setup.py install > /dev/null 2>&1

mnist: ## Run MNIST example
	@python examples/mnist.py