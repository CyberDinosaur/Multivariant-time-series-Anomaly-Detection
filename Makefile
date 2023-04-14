.PHONY: clean, sync

clean:
	rm -rf outputs

sync:
	git push -u origin main
