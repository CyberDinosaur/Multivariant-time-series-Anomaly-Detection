.PHONY: clean, sync

clean:
	rm -rf outputs

sync:
	git push origin main
