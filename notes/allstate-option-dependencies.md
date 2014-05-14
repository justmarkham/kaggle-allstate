## Allstate Option Dependencies

As discussed in my [paper](../allstate-paper.md) in Data Exploration section 6, I compiled a loose set of "rules" that attempted to capture dependencies between different options. In Model Building section 9 of the paper, I used these rules to manually "fix" the predicted plan for customers that my model predicted had a strong likelihood of changing options between their final quote and the purchase point.

* Option A (0, 1, 2)
	* if B=1 or C=3 or D=3 or E=1 or F=1/2, more likely to choose A=1
	* if E=1, almost never choose A=0
	* if F=0, more likely to choose A=0
	* if F=3, never choose A=1
* Option B (0, 1)
	* if A=0 or C=1 or E=0 or F=0, more likely to choose B=0
	* if E=1, more likely to choose B=1
* Option C (1, 2, 3, 4)
	* if A=1 or D=3 or E=1 or F=1, more likely to choose C=3
	* if D=1, almost always choose C=1
	* if D=2, never choose C=4
* Option D (1, 2, 3)
	* if A=1 or C=3, more likely to choose D=3
	* if C=2/3, almost never choose D=1
	* if C=4, always choose D=3
* Option E (0, 1)
	* if A=0, almost always choose E=0
	* if B=0 or C=1 or F=0, more likely to choose E=0
* Option F (0, 1, 2, 3)
	* if A=0, almost always choose F=0
	* if A=1, never choose F=3
* Option G (1, 2, 3, 4)
	* no patterns
