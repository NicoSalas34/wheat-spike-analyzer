# Data

## Structure

```
data/
├── test_sample/      # 3 images de test incluses dans le repo
├── raw/              # Images brutes à analyser (non versionné, .gitignore)
└── validation/       # Images de validation (non versionné)
```

## Test rapide

```bash
python src/main.py data/test_sample/ --batch --low-debug
```

## Images brutes

Placez vos images (JPG/PNG) dans `data/raw/` pour l'analyse en batch :

```bash
python src/main.py data/raw/ --batch --resume --low-debug
```
