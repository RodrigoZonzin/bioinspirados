push: 
	git add .
	git commit -m 'commit $(shell date +%d-%m-%Y) by $(USER)'
