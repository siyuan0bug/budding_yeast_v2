"""Compact workflow summaries; all trajectories remain in their source pools."""
import csv
import os
from pathlib import Path
from budding_yeast_v2.utils.experiments.curves import all_rounds_figure,performance_figure


def save_progress_reports(directory,records,histories,contexts,round_dir):
    directory=Path(directory)
    all_rounds_figure(histories,directory/'loss.png',contexts)
    performance_figure(records,directory/'performance.png')
    with (directory/'performance.csv').open('w',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(('round','training_rows','domain','metric_version',
                         'Relative L2','macro_trajectory_rel_l2','MAE','MSE','Correlation'))
        for row in records:
            for domain,values in row['metrics'].items():
                m=values['all_variables']
                writer.writerow((row['round'],row['training_rows'],domain,values.get('metric_version','historical_unspecified'),
                                 m['Relative L2'],values['macro_trajectory_rel_l2'],m['MAE'],m['MSE'],m['Correlation']))
    # Atomic link update, no copies of final figures, CSVs or trajectories.
    temporary=directory/'._eval_next'
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(os.path.relpath(Path(round_dir)/'eval',directory),target_is_directory=True)
    os.replace(temporary,directory/'eval')
