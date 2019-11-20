#!/bin/bash


num_jobs=32
root_data_dir="/data/icecube"
histo_data_dir="~/data_mc_agreement"
histo_plot_dir="~/data_mc_agreement"

declare -a pulse_serieses=( \
    "SRTTWOfflinePulsesDC" \
    "SRTInIcePulses" \
    "SplitInIcePulses" \
    )

#declare -a z_regionses=( \
#    "0 1 2" \
#    )

declare -a filter_keyses=( \
    "--qntm=0.05               --min-pulse-q=0  " \
    "--qntm=0.05               --min-pulse-q=0    --coinc-mu-filt" \
    "--qntm=0.05               --min-pulse-q=0.2" \
    "--qntm=0.05               --min-pulse-q=0.2  --coinc-mu-filt" \
    "--qntm=0.05               --min-pulse-q=0.25" \
    "--qntm=0.05               --min-pulse-q=0.25 --coinc-mu-filt" \
    "--qntm=0.05               --min-pulse-q=0.3" \
    "--qntm=0.05               --min-pulse-q=0.3  --coinc-mu-filt" \
    \
    "--qntm=0.05 --min-evt-p=8 --min-pulse-q=0 " \
    "--qntm=0.05 --min-evt-p=8 --min-pulse-q=0    --coinc-mu-filt" \
    "--qntm=0.05 --min-evt-p=8 --min-pulse-q=0.2" \
    "--qntm=0.05 --min-evt-p=8 --min-pulse-q=0.2  --coinc-mu-filt" \
    "--qntm=0.05 --min-evt-p=8 --min-pulse-q=0.25" \
    "--qntm=0.05 --min-evt-p=8 --min-pulse-q=0.25 --coinc-mu-filt" \
    "--qntm=0.05 --min-evt-p=8 --min-pulse-q=0.3" \
    "--qntm=0.05 --min-evt-p=8 --min-pulse-q=0.3  --coinc-mu-filt" \
    \
    "--qntm=0.05 --min-evt-d=8 --min-pulse-q=0 " \
    "--qntm=0.05 --min-evt-d=8 --min-pulse-q=0    --coinc-mu-filt" \
    "--qntm=0.05 --min-evt-d=8 --min-pulse-q=0.2" \
    "--qntm=0.05 --min-evt-d=8 --min-pulse-q=0.2  --coinc-mu-filt" \
    "--qntm=0.05 --min-evt-d=8 --min-pulse-q=0.25" \
    "--qntm=0.05 --min-evt-d=8 --min-pulse-q=0.25 --coinc-mu-filt" \
    "--qntm=0.05 --min-evt-d=8 --min-pulse-q=0.3" \
    "--qntm=0.05 --min-evt-d=8 --min-pulse-q=0.3  --coinc-mu-filt" \
    )

declare -a set_keys=( \
    "(12, (1, 3))" \
    "(13, (1, 3))" \
    "(14, (1, 3))" \
    "(15, (1, 3))" \
    "(16, (1, 3))" \
    "(17, (1, 3))" \
    "(18, (1, 3))" \
    "(139011, (1, 3))" \
    "(888003, (1, 3))" \
    "(120000, (1, 3))" \
    "(140000, (1, 3))" \
    "(160000, (1, 3))" \
    )

declare -a mc_setses=( \
    "baseline" \
    )


t0=$( date +'%Y-%m-%dT%H%M%z' )

for pulse_series in "${pulse_serieses[@]}" ; do
    #for z_regions in "${z_regionses[@]}" ; do
        for filter_keys in "${filter_keyses[@]}" ; do
            for set_key in "${set_keys[@]}" ; do
                while (( $( jobs -r | wc -l ) >= ${num_jobs} )) ; do sleep 0.4 ; done
                ~/src/retro/retro/utils/data_mc_agreement__make_histos.py populate \
                    --root-data-dir "$root_data_dir" \
                    --histo-data-dir "$histo_data_dir" \
                    --set-key "$set_key" \
                    --pulse-series "$pulse_series" \
                    $filter_keys \
                    &
                    #--z-regions $z_regions
            done
        done
    #done
done

wait


for pulse_series in "${pulse_serieses[@]}" ; do
    #for z_regions in "${z_regionses[@]}" ; do
        for filter_keys in "${filter_keyses[@]}" ; do
            for mc_set in "${mc_setses[@]}" ; do
                while (( $( jobs -r | wc -l ) >= ${num_jobs} )) ; do sleep 0.4 ; done
                ~/src/retro/retro/utils/data_mc_agreement__make_histos.py plot \
                    --root-data-dir "$root_data_dir" \
                    --histo-data-dir "$histo_data_dir" \
                    --histo-plot-dir "$histo_plot_dir" \
                    --mc-set "$mc_set" \
                    --pulse-series "$pulse_series" \
                    $filter_keys \
                    &
                    #--z-regions $z_regions
            done
        done
    #done
done

wait


echo "$t0 --> $( date +'%Y-%m-%dT%H%M%z' )"
