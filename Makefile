.PHONY: clean, sync

clean:
	rm -rf outputs
sync:
	git push -u MAD main
run:
    ./run.sh
