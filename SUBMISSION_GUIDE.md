Submission Guide — Banking Dataset Project (Group 1)

Purpose
- This repository contains a complete CRISP-DM analysis for the `banking_realistic.csv` dataset.
- Upload the repository to GitHub and open it in Posit Cloud (recommended) to run the analysis without local package installs.

Included files (key)
- `Project_datasets/` : dataset files (includes `banking_realistic.csv`).
- `analysis/banking_project_full.R` : full R script (CRISP-DM) — run this to perform the whole analysis.
- `analysis/project_report.md` : project report in Markdown (ready to convert to DOCX/PDF).
- `analysis/banking_analysis.Rmd` : RMarkdown starter notebook for interactive use in Posit.
- `analysis/packages.R` : helper to install required R packages.
- `analysis/run_all.R` : small wrapper to run the analysis script.
- `analysis/README.md` : quick instructions for running locally / Posit Cloud.

Quick steps — recommended (Posit Cloud)
1. Push this repository to GitHub (see commands below) or zip the repo and upload.
2. In Posit Cloud: New Project -> From Git -> paste GitHub repo URL. Or New Project -> Upload files and upload this repo.
3. Open `analysis/run_all.R` or `analysis/banking_project_full.R` in the Posit editor.
4. Run `analysis/packages.R` first to install packages (or run `run_all.R` which sources it).
5. Run `banking_project_full.R` (Source or Run) — outputs will be saved in `analysis/outputs/` and `analysis/plots/`.

Git commands (optional)
```bash
git add .
git commit -m "Add complete banking dataset analysis for submission"
git push origin main
```

Convert Markdown to Word (optional)
- In Posit Cloud you can open `analysis/project_report.md` and use the Knit / Render option to produce HTML/PDF/Word depending on your setup.
- Or, locally / in Posit terminal, run:
```bash
pandoc analysis/project_report.md -o analysis/project_report.docx --from markdown -s
```

Checklist before submission
- [ ] Add student names in `analysis/banking_project_full.R` header.
- [ ] Verify `Project_datasets/banking_realistic.csv` is present.
- [ ] Run `analysis/run_all.R` in Posit Cloud and ensure outputs created in `analysis/outputs`.
- [ ] Export `analysis/project_report.md` to `.docx` if the instructor requires Word.

If you want, I can also prepare a ready-to-upload ZIP of the repository contents — tell me and I will create it here.
