# Views

KempnerPulse has four interactive views. Switch between them with the
`:`-commands while the dashboard runs (see {doc}`../guide/cli`).

![KempnerPulse demo](../images/kempner_pulse_screen_record.gif)

## Fleet view

All GPUs at a glance — Real Utilization, memory, power, temperature, PCIe/NVLink
bandwidth, and sparkline bars. This is the default view.

![Fleet view](../images/fleet_view.png)

## Focus view

One GPU in depth, with a per-metric table and sparkline history. Enter with
`:focus <id>`.

![Focus view](../images/focus_view.png)

## Plot view

Line charts of each metric across all GPUs over the recent history window. Enter
with `:plot`.

![Plot view](../images/plot_view.png)

## Job view

The running GPU compute processes, with the per-GPU status/utilization for each.
Enter with `:job`.

![Job view](../images/job_view.png)
