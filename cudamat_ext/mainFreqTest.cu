#include <stdio.h>
#include <cuda.h>
#include "FrequencyTemplates_kernels.cuh"

const int NUM_THREADS = 32;
void ComputeSpectrogram(float *template_values, 
                           float *template_parameters,
                           float * (output_spectrograms),
                           float sampling_frequency,
                           int num_samples_per_spectrogram,
                           int num_formants, 
                           int num_F0_harmonics,
                           int num_template_parameters, 
                           int num_frames)
{
   int numThreadsPerBlock = NUM_THREADS ;  
   float freq_width = sampling_frequency / num_samples_per_spectrogram ;
   int num_template_values = 2+ 2*num_formants + 2 ; 

   if (num_template_parameters != num_F0_harmonics + 2 + 2*num_formants + 1)
      throw "Number of template parameters in template_parameters \
             is inconsistent with arguments";


   int spec_dim = num_samples_per_spectrogram/2 + 1 ;
   float *dPtr_template_values, *dPtr_template_parameters, *dPtr_output_spectrograms ;
   cudaMalloc((void **) &dPtr_template_values, 
              num_frames * num_template_values * sizeof(float)) ;
   cudaMalloc((void **) &dPtr_template_parameters, 
              num_template_parameters * numThreadsPerBlock * sizeof(float)) ;
   cudaMalloc((void **) &dPtr_output_spectrograms, 
              num_frames * spec_dim * sizeof(float)) ;

   cudaMemcpy(dPtr_template_values,
              template_values, 
              num_frames * num_template_values * sizeof(float),
              cudaMemcpyHostToDevice); 
   cudaMemcpy(dPtr_template_parameters,
              template_parameters, 
              num_template_parameters * numThreadsPerBlock * sizeof(float),
              cudaMemcpyHostToDevice); 

   kSpeechTemplates<<<num_frames, numThreadsPerBlock>>>(
                                                dPtr_template_values,
                                                dPtr_template_parameters,
                                                dPtr_output_spectrograms,
                                                freq_width, 
                                                spec_dim,
                                                num_F0_harmonics,
                                                num_formants, 
                                                num_template_parameters, 
                                                num_template_values);
   cudaThreadSynchronize();
   cudaMemcpy(output_spectrograms, dPtr_output_spectrograms, 
              sizeof(float)*num_frames*spec_dim,
              cudaMemcpyDeviceToHost);  

   cudaFree(dPtr_output_spectrograms);  
   cudaFree(dPtr_template_values);  
   cudaFree(dPtr_template_parameters);  
}


void GetTemplateParameters1(float * (&template_parameters), int num_F0_harmonics)
{

   float amp_ratio_harmonics = 0.8;
   template_parameters[0] = 1.;
   for (int harmonic_num = 1 ; harmonic_num < num_F0_harmonics ; harmonic_num++) 
   {
      template_parameters[harmonic_num] = template_parameters[harmonic_num-1]*amp_ratio_harmonics;
   }

   int index = num_F0_harmonics;
   float sigma = 20.0 ;
   template_parameters[index++] = 100;
   template_parameters[index++] = 1.0/(2*sigma*sigma) ;

   // Formants for sound a, speaker 1 in praat.
   // 1st formant
   sigma = 50.0 ;
   template_parameters[index++] = 720;
   template_parameters[index++] = 1.0/(2*sigma*sigma) ;

   // 2nd formant
   sigma = 200.0 ;
   template_parameters[index++] = 1060 ;
   template_parameters[index++] = 1./(2*sigma*sigma) ;

   // 3rd formant
   sigma = 300.0 ;
   template_parameters[index++] = 2420 ;
   template_parameters[index++] = 1./(2*sigma*sigma) ;

   // fricative threshold
   template_parameters[index++] = 3400 ;
 
}
void GetTemplateValues1(float * (&template_values), int num_formants)
{
   int index = 0;

   // pitch relative frequency
   template_values[index++] = 1.0;
   // pitch amplitude
   template_values[index++] = 1.0 ; 

   if (index/2-1 < num_formants)
   {
      // first formant relative frequency
      template_values[index++] = 1.0;
      // first formant amplitude
      template_values[index++] = 0.5 ; 
   }

   if (index/2-1 < num_formants)
   {
      // second formant relative frequency
      template_values[index++] = 1.0;
      // second formant amplitude
      template_values[index++] = 0.25 ; 
   }

   if (index/2-1 < num_formants)
   {
      // third formant relative frequency
      template_values[index++] = 1.0;
      // third formant amplitude
      template_values[index++] = 0.1 ; 
   }

   // fricative relative frequency
   template_values[2*num_formants+2] = 1.0;
   // fricative relative amplitude
   template_values[2*num_formants+3] = .2;

}

void GetTemplateValues(float * (&template_values), 
                       int num_formants, 
                       int num_template_values, 
                       int num_frames)
{
   float min_pitch = 0.5 ; 
   float max_pitch = 1.0;

   for (int frame_num = 0 ; frame_num < num_frames ; frame_num++) 
   {
      int index = frame_num* num_template_values;

      // pitch relative frequency
      float pitch_value = 0;
      if (frame_num < num_frames/2)
         pitch_value = min_pitch + ((max_pitch-min_pitch) * frame_num * 2)/num_frames ; 
      else
         pitch_value = max_pitch - ((max_pitch-min_pitch) * (frame_num-num_frames/2)*2)/num_frames ; 

      printf("%f\n", pitch_value);
      template_values[index++] = 1.0 * pitch_value ; 
      // pitch amplitude
      template_values[index++] = 1.0 ; 
   
      // first formant relative frequency
      template_values[index++] = pitch_value;
      // first formant amplitude
      template_values[index++] = 0.5 ; 
   
      // second formant relative frequency
      template_values[index++] = 1 ;
      // second formant amplitude
      template_values[index++] = 0.25 ; 
   
      // third formant relative frequency
      template_values[index++] = pitch_value;
      // third formant amplitude
      template_values[index++] = 0.1 ; 
   
      // fricative relative frequency
      template_values[index++] = pitch_value;
      // fricative relative amplitude
      template_values[index++] = .2;
      if(index != (frame_num+1)*num_template_values)
         throw (const char *) "Unexpected numbers";
   }

}

void ReplicateTemplateParameters(float *template_parameters, 
                                 int num_template_parameters,
                                 int num_replications)
{
   for (int replication_num = 0 ; replication_num < num_replications-1 ; 
        replication_num++) 
   {
      memcpy(&template_parameters[(replication_num+1)*num_template_parameters], 
             template_parameters, 
             num_template_parameters*sizeof(float));

   }
}

void TestDecodeSpec()  
{ 
   int num_frames = 10;
   int num_samples_per_spectrogram = 1024;
   int spec_dim = num_samples_per_spectrogram/2+1;
   int sampling_frequency = 16000 ;
   int num_formants = 3 ;
   int num_F0_harmonics = 25 ;
   int num_template_parameters = num_F0_harmonics + 2 + 2*num_formants + 1 ;
   int num_template_values = 2 + 2*num_formants + 2 ;

   float *template_parameters = new float[num_template_parameters*NUM_THREADS] ; 
   float *template_values = new float [num_template_values*num_frames];
   float *output_spectrograms = new float[spec_dim*num_frames];

   GetTemplateParameters1(template_parameters, num_F0_harmonics); 
   ReplicateTemplateParameters(template_parameters, num_template_parameters, NUM_THREADS); 
   //GetTemplateValues1(template_values, num_formants); 
   GetTemplateValues(template_values, num_formants, num_template_values, num_frames); 
   
   ComputeSpectrogram(template_values, 
                         template_parameters,
                         output_spectrograms,
                         sampling_frequency,
                         num_samples_per_spectrogram,
                         num_formants, 
                         num_F0_harmonics,
                         num_template_parameters,
                         num_frames);

   bool blnPrintResults = true ; 
   if (blnPrintResults == true)
   {
      FILE *fpParams = fopen("template_parameters.txt", "w"); 
      for (int j=0 ; j< num_template_parameters ; j++)
      {
         fprintf(fpParams, "%f", template_parameters[j]) ;
         if (j != num_template_parameters-1)
            fprintf(fpParams, "\t") ; 
      }
      fprintf(fpParams, "\n") ; 
      fclose(fpParams) ; 

      FILE *fpValues = fopen("template_values.txt", "w"); 
      for (int frame_num = 0 ; frame_num < num_frames ; frame_num++ ) 
      {
         for (int j=0 ; j< num_template_values ; j++)
         {
            fprintf(fpValues, "%f", template_values[frame_num*num_template_values+j]) ;
            if (j != num_template_values-1)
               fprintf(fpValues, "\t") ; 
         }
         fprintf(fpValues, "\n") ; 
      }
      fclose(fpValues) ; 

      FILE *fpSpec = fopen("spectrograms.txt", "w"); 
      for (int frame_num = 0 ; frame_num < num_frames ; frame_num++ ) 
      {
         for (int j=0 ; j< spec_dim ; j++)
         {
            fprintf(fpSpec, "%f", output_spectrograms[frame_num*spec_dim+j]) ;
            if (j != spec_dim-1)
               fprintf(fpSpec, "\t") ; 
         }
         fprintf(fpSpec, "\n") ; 
      }
      fclose(fpSpec) ; 
   }

   // Cleanup  
   delete [] template_parameters ; 
   delete [] template_values ; 
   delete [] output_spectrograms ; 

} 

int main(int argc, char **argv)
{
   TestDecodeSpec() ; 
   cudaThreadExit();
}

