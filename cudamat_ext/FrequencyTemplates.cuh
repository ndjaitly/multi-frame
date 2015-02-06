#pragma once
extern int ComputeSpectrogram(cudamat *template_values, 
                              cudamat *template_parameters,
                              cudamat *output_spectrograms,
                              float sampling_frequency,
                              int num_samples_per_spectrogram,
                              int num_formants, 
                              int num_F0_harmonics);

extern int ComputeSpectrogramGradients(cudamat *template_values, 
                              cudamat *template_parameters,
                              cudamat *spectrogram_diffs,
                              cudamat *template_values_gradients,
                              cudamat *template_parameters_gradients,
                              float sampling_frequency,
                              int num_samples_per_spectrogram,
                              int num_formants, 
                              int num_F0_harmonics);

extern int ComputeSpectrogramNonParametric(cudamat *template_values, 
                              cudamat *template_parameters,
                              cudamat *output_spectrograms);

extern int ComputeSpectrogramGradientsNonParametric(
                              cudamat *template_values, 
                              cudamat *template_parameters,
                              cudamat *spectrogram_diffs,
                              cudamat *template_values_gradients,
                              cudamat *template_parameters_gradients);
