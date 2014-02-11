/*
 * Copyright 2013 Olivier Caron-Lizotte
 * olivierlizotte@gmail.com
 * Licensed under the MIT license: <http://www.opensource.org/licenses/mit-license.php>
 */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace PeptidAce.GPU
{
    /// <summary>
    /// Object returned with results
    /// </summary>
    public class ProteinPrecursorMatch
    {
        public int proteinStartPos;
        public int proteinEndPos;
        public int firstQueryIndex;
        public ProteinPrecursorMatch(int start, int stop, int index)
        {
            proteinStartPos = start;
            proteinEndPos = stop;
            firstQueryIndex = index;
        }
    }

    /// <summary>
    /// This class is used to digest, inGraphico, a protein sequence (as a set of weights) into peptide sequences (as a starting point and a length)
    /// Will only report peptides matching a list of given masses
    /// TODO transform the protein mass array into a matrix to have the possibility of including variable modifications
    /// </summary>
    public class ProteinDigest
    {
        //Cudafy gpu object
        private GPGPU gpu;
        
        private int maxGridSize;
        private int peptideArraySize;
        private int maxPeptideSize;
        private int minPeptideSize;
        private double[] dev_prec;
        private int[] dev_outputStart;
        int[] outputStart;
        public ProteinDigest(double[] potentialPrecursors, int maxPeptideLength, int minPeptideLength)
        {
            //Init Gpu access
            CudafyModule km = CudafyTranslator.Cudafy();

            gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            gpu.LoadModule(km);
            dev_prec = gpu.CopyToDevice(potentialPrecursors);

            // allocate the memory on the GPU
            GPGPUProperties properties = gpu.GetDeviceProperties();
            this.maxGridSize = properties.MaxGridSize.x;
            this.maxPeptideSize = maxPeptideLength;
            this.peptideArraySize = maxPeptideLength - minPeptideLength + 1;
            this.minPeptideSize = minPeptideLength;
            this.outputStart = new int[maxGridSize * peptideArraySize];

            //Allocate vector that will store the results
            dev_outputStart = gpu.Allocate<int>(maxGridSize * peptideArraySize);
        }

        public void Dispose()
        {
            gpu.FreeAll();
        }

        public IEnumerable<ProteinPrecursorMatch> Execute(double[] aminoAcidSequenceMasses, double precisionInPPM, double maxWeight)
        {
            try
            {
                // copy the arrays to the GPU
                double[] dev_prot = gpu.Allocate<double>(aminoAcidSequenceMasses);
                gpu.CopyToDevice(aminoAcidSequenceMasses, dev_prot);

                if (maxGridSize < aminoAcidSequenceMasses.Length)
                    Console.WriteLine("GPU not supporting protein length above " + maxGridSize);

                double precisionInPPMDivided = precisionInPPM / 1e6;
                int size = 1 + aminoAcidSequenceMasses.Length / 32;
                
                //Launch the spectrum matching AND noenzyme in silico protein digest on the GPU
                gpu.Launch(size, 32).matchSpectrum(dev_prot, dev_prec, dev_outputStart, precisionInPPMDivided, maxWeight, peptideArraySize, minPeptideSize);
                
                //Make sure everything was executed before going forward
                gpu.Synchronize();

                //Free the protein mass array
                gpu.Free(dev_prot);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.WriteLine(ex.StackTrace);
            }

            //Return every match one at a time, by yielding the Protein Precursor Match object
            for (int i = 0; i < aminoAcidSequenceMasses.Length; i++)
            {
                gpu.CopyFromDevice(dev_outputStart, i * peptideArraySize, outputStart, 0, peptideArraySize);
                for (int j = 0; j < peptideArraySize; j++)
                {
                    if (outputStart[j] >= 0)
                        yield return new ProteinPrecursorMatch(i, i + j + minPeptideSize - 1, outputStart[j]);
                }
            }//*/
        }

        [Cudafy]
        public static void matchSpectrum(GThread thread, double[] dev_prot, double[] dev_prec, int[] dev_outputStart, double precisionInPPMDivided, double maxWeight, int peptideArraySize, int minPeptideSize)
        {
            int aminoAcidIndex = (thread.threadIdx.x + thread.blockIdx.x * 32);
            int aminoAcidCumulIndex = aminoAcidIndex;
            int precursorIndex = 0;

            int peptideLength = 0;
            if (aminoAcidIndex < dev_prot.Length)
            {
                int outputIndex = aminoAcidIndex * peptideArraySize;
                for (int i = 0; i < peptideArraySize; i++)
                    dev_outputStart[outputIndex + i] = -1;

                double cumulMassProt = 0.0;
                double massProtTolerance;
                while (peptideLength <= peptideArraySize + minPeptideSize && aminoAcidCumulIndex < dev_prot.Length && precursorIndex < dev_prec.Length)
                {
                    cumulMassProt += dev_prot[aminoAcidCumulIndex];
                    peptideLength++;

                    massProtTolerance = cumulMassProt * precisionInPPMDivided;
                    while (precursorIndex < dev_prec.Length-1 && dev_prec[precursorIndex] < cumulMassProt - massProtTolerance)
                        precursorIndex++;

                    if (peptideLength - minPeptideSize >= 0 && dev_prec[precursorIndex] >= cumulMassProt - massProtTolerance && dev_prec[precursorIndex] <= cumulMassProt + massProtTolerance && dev_outputStart[outputIndex + peptideLength - minPeptideSize] == -1)
                    {
                        dev_outputStart[outputIndex + peptideLength - minPeptideSize] = precursorIndex;
                        precursorIndex++;
                    }

                    aminoAcidCumulIndex++;
                }
            }
        }
    }
}
