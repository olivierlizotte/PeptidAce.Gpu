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

namespace Trinity_Gpu
{
    /// <summary>
    /// Object returned with results
    /// </summary>
    public class ProteinPrecursorMatch
    {
        public int proteinStartPos;
        public int proteinEndPos;
        public int firstQueryIndex;
    }

    /// <summary>
    /// This class is used to digest, inGraphico, a protein sequence (as a set of weights) into peptide sequences (as a starting point and a length)
    /// Will only report peptides matching a list of given masses
    /// TODO transform the protein mass array into a matrix to have the possibility of including variable modifications
    /// </summary>
    public class ProteinDigest
    {
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
                //double[] dev_prot = gpu.CopyToDevice(aminoAcidSequenceMasses);

                // Launch 128 blocks of 128 threads each
                if (maxGridSize < aminoAcidSequenceMasses.Length)
                    Console.WriteLine("GPU not supporting protein length above " + maxGridSize);

                double precisionInPPMDivided = precisionInPPM / 1e6;
                int size = 1 + aminoAcidSequenceMasses.Length / 32;
                //int size = 1 + 1234 / 32;
                gpu.Launch(size, 32).matchSpectrum(dev_prot, dev_prec, dev_outputStart, precisionInPPMDivided, maxWeight, peptideArraySize, minPeptideSize);
                //matchSpectrum(null, aminoAcidSequenceMasses, potentialPrecursors, outputStart, outputEnd, aminoAcidSequenceMasses.Length, potentialPrecursors.Length, precision, maxWeight);                
                /*
                if (gpu.IsOnGPU(dev_outputStart))
                    gpu.CopyFromDevice(dev_outputStart, 0, outputStart, 0, aminoAcidSequenceMasses.Length * maxPeptideLength);
                else
                    Console.WriteLine("B@#$@#$T");//*/
                //gpu.CopyFromDevice(dev_outputEnd, outputEnd);
                gpu.Synchronize();
                gpu.Free(dev_prot);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.WriteLine(ex.StackTrace);
            }
            //return outputStart;

            //int[] test = new int[maxPeptideLength];
            ProteinPrecursorMatch result = new ProteinPrecursorMatch();            
            for (int i = 0; i < aminoAcidSequenceMasses.Length; i++)
            {
                gpu.CopyFromDevice(dev_outputStart, i * peptideArraySize, outputStart, 0, peptideArraySize);
                for (int j = 0; j < peptideArraySize; j++)
                {
                    if (outputStart[j] >= 0)//i + j * maxPeptideLength] >= 0)
                    {
                        result.proteinStartPos = i;// outputStart[i] + 1;
                        result.proteinEndPos = i + j + minPeptideSize - 1;
                        result.firstQueryIndex = outputStart[j];//i + j * maxPeptideLength];
                        yield return result;
                    }
                }
            }//*/
        }
        /*
        [Cudafy]
        public static void test(GThread thread, double[] dev_prot, int[] dev_outputStart)
        {
            int tid = thread.threadIdx.x;// +thread.blockIdx.x * thread.blockDim.x;
            //int aminoAcidIndex = 13;// thread.gridDim.x;
            //int precursorIndex = thread.blockIdx.x * thread.blockDim.x;
            if (tid == 0)
            {
                dev_outputStart[0] = tid;
                dev_outputStart[1] = thread.blockIdx.x;
                dev_outputStart[2] = thread.blockDim.x;
                dev_outputStart[3] = thread.threadIdx.x;
                dev_outputStart[4] = thread.threadIdx.x;
                dev_outputStart[5] = 13;
            }
        }//*/
        /*
        public static void matchSpectrum(GThread thread, double[] dev_prot, double[] dev_prec, int[] dev_outputStart, int[] dev_outputEnd, int nbProt, int nbPrec, double precision, double maxWeight)
        {
            for (int aminoAcidIndex = 0; aminoAcidIndex < nbProt; aminoAcidIndex++ )
            {
                int aminoAcidCumulIndex = aminoAcidIndex;
                double cumulMassProt = 0.0;
                while (aminoAcidCumulIndex < nbProt && cumulMassProt < maxWeight)
                {
                    cumulMassProt += dev_prot[aminoAcidCumulIndex];
                    for (int i = 0; i < nbPrec; i++)
                    {
                        if (dev_prec[i] >= cumulMassProt - precision && dev_prec[i] <= cumulMassProt + precision)
                        {
                            dev_outputStart[i] = aminoAcidIndex;
                            dev_outputEnd[i] = aminoAcidCumulIndex;
                        }
                    }
                    aminoAcidCumulIndex++;
                }
            }
        }//*/

        // Does not return all possible mixes!
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

        [Cudafy]
        public static void checkPeptide(GThread thread, double[] dev_prot, double[] dev_prec, int[] dev_outputStart, int[] dev_outputEnd, int nbProt, int nbPrec, double precision, double maxWeight)
        {
            int aminoAcidIndex = thread.blockIdx.x;
            int precursorIndex = thread.threadIdx.x;

            int aminoAcidCumulIndex = aminoAcidIndex;
            double cumulMass = 0.0;
            while (aminoAcidCumulIndex < nbProt && cumulMass < maxWeight)
            {
                cumulMass += dev_prot[aminoAcidCumulIndex];
                if ((dev_prec[precursorIndex] > cumulMass && dev_prec[precursorIndex] - precision <= cumulMass) ||
                   (dev_prec[precursorIndex] < cumulMass && dev_prec[precursorIndex] + precision >= cumulMass))
                {
                    dev_outputStart[precursorIndex] = aminoAcidIndex;
                    dev_outputEnd[precursorIndex] = aminoAcidCumulIndex;
                }
                aminoAcidCumulIndex++;
            }
        }
        //*/

        /*         
        [Cudafy]
        public static void checkPeptide(GThread thread, double[] dev_prot, double[] dev_prec, int[] dev_outputStart, int[] dev_outputEnd, int nbProt, int nbPrec, double precision, double maxWeight, int proteinStartPos)
        {
            int aminoAcidIndex = proteinStartPos;
            int precursorIndex = thread.blockDim.x;

            int aminoAcidCumulIndex = aminoAcidIndex;
            double cumulMass = 0.0;
            while (aminoAcidCumulIndex < nbProt || cumulMass > maxWeight)
            {
                cumulMass += dev_prot[aminoAcidCumulIndex];
                if ((dev_prec[precursorIndex] > cumulMass && dev_prec[precursorIndex] - precision <= cumulMass) ||
                   (dev_prec[precursorIndex] < cumulMass && dev_prec[precursorIndex] + precision >= cumulMass))
                {
                    dev_outputStart[precursorIndex] = aminoAcidIndex;
                    dev_outputEnd[precursorIndex] = aminoAcidCumulIndex;
                }
            }
        }//*/
    }
}
