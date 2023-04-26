all: kill clean build run 
		
build: build_base build_gitpod

build_gitpod:
	docker tag jcabrero/flp:latest jcabrero/flp:gitpod

build_base:
	docker build --network=host \
	-f Dockerfile -t jcabrero/flp:latest ..

flp:
	docker run  \
		--net host \
		--hostname=$@ --name=$@ \
		-it --rm \
		jcabrero/flp:latest bash

run: flp 

clean:
	docker rmi $(shell docker images --filter dangling=true -q --no-trunc) || true

kill:
	docker kill flp || true

