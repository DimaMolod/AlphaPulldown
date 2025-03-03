"""
Running the test script:
1. Batch job on gpu-el8
sbatch test_predict_structure.sh

2. Interactive session on gpu-el8
salloc -p gpu-el8 --ntasks 1 --cpus-per-task 8 --qos=highest --mem=16000 -C gaming -N 1 --gres=gpu:1 -t 05:00:00 

module load Anaconda3 
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1
conda activate AlphaPulldown
srun python test/check_predict_structure.py # this will be slower due to the slow compilation error

"""
import shutil
import tempfile
import unittest
import sys
import os
import subprocess
import json

import alphapulldown
from alphapulldown import predict_structure

FAST=True
if FAST:
    from alphafold.model import config
    config.CONFIG_MULTIMER.model.embeddings_and_evoformer.evoformer_num_block = 1
    #TODO: can it be done faster? For P0DPR3_and_P0DPR3 example, I think most of the time is taken by jax model compilation.

class _TestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.data_dir = "/scratch/AlphaFold_DBs/2.3.2/"
        #Get test_data directory as relative path to this script
        self.test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")

class TestScript(_TestBase):
    #Add setup that creates empty output directory temporary
    def setUp(self) -> None:
        #Call the setUp method of the parent class
        super().setUp()

        #Create a temporary directory for the output
        self.output_dir = tempfile.mkdtemp()
        self.test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
        self.protein_lists = os.path.join(self.test_data_dir, "tiny_monomeric_features_homodimer.txt")
        self.monomer_objects_dir = self.test_data_dir

        #Get path of the alphapulldown module
        alphapulldown_path = alphapulldown.__path__[0]
        #join the path with the script name
        self.script_path = os.path.join(alphapulldown_path, "run_multimer_jobs.py")
        print(sys.executable)
        print(self.script_path)
        self.args = [
            sys.executable,
            self.script_path,
            "--mode=custom",
            "--num_cycle=1",
            "--num_predictions_per_model=1",
            f"--output_path={self.output_dir}",
            f"--data_dir={self.data_dir}",
            f"--protein_lists={self.protein_lists}",
            f"--monomer_objects_dir={self.monomer_objects_dir}"
        ]

    def tearDown(self) -> None:
        #Remove the temporary directory
        shutil.rmtree(self.output_dir)

    def _runCommonTests(self, result):
        print(result.stdout)
        print(result.stderr)
        self.assertEqual(result.returncode, 0, f"Script failed with output:\n{result.stdout}\n{result.stderr}")
        #Get the name of the first directory in the output directory
        dirname = os.listdir(self.output_dir)[0]
        #Check if the directory contains five files starting from ranked and ending with .pdb
        self.assertEqual(len([f for f in os.listdir(os.path.join(self.output_dir, dirname)) if f.startswith("ranked") and f.endswith(".pdb")]), 5)
        #Check if the directory contains five files starting from result and ending with .pkl
        self.assertEqual(len([f for f in os.listdir(os.path.join(self.output_dir, dirname)) if f.startswith("result") and f.endswith(".pkl")]), 5)
        #Check if the directory contains five files ending with png
        self.assertEqual(len([f for f in os.listdir(os.path.join(self.output_dir, dirname)) if f.endswith(".png")]), 5)
        #Check if the directory contains ranking_debug.json
        self.assertTrue("ranking_debug.json" in os.listdir(os.path.join(self.output_dir, dirname)))
        #Check if the directory contains timings.json
        self.assertTrue("timings.json" in os.listdir(os.path.join(self.output_dir, dirname)))
        #Check timings_temp.json is not present
        self.assertFalse("timings_temp.json" in os.listdir(os.path.join(self.output_dir, dirname)))
        #Check if all files not empty
        for f in os.listdir(os.path.join(self.output_dir, dirname)):
            self.assertGreater(os.path.getsize(os.path.join(self.output_dir, dirname, f)), 0)
        #open the ranking_debug.json file and check if the number of models is 5
        with open(os.path.join(self.output_dir, dirname, "ranking_debug.json"), "r") as f:
            ranking_debug = json.load(f)
            self.assertEqual(len(ranking_debug["order"]), 5)
            if "iptm+ptm" in ranking_debug:
                expected_set = set(["model_1_multimer_v3_pred_0", "model_2_multimer_v3_pred_0", "model_3_multimer_v3_pred_0", "model_4_multimer_v3_pred_0", "model_5_multimer_v3_pred_0"])
                self.assertEqual(len(ranking_debug["iptm+ptm"]), 5)
                #Check if order contains the correct models
                self.assertSetEqual(set(ranking_debug["order"]), expected_set)
                #Check if iptm+ptm contains the correct models
                self.assertSetEqual(set(ranking_debug["iptm+ptm"].keys()), expected_set)
            elif "plddt" in ranking_debug:
                expected_set = set(["model_1_pred_0", "model_2_pred_0", "model_3_pred_0", "model_4_pred_0", "model_5_pred_0"])
                self.assertEqual(len(ranking_debug["plddt"]), 5)
                self.assertSetEqual(set(ranking_debug["order"]), expected_set)
                #Check if iptm+ptm contains the correct models
                self.assertSetEqual(set(ranking_debug["plddt"].keys()), expected_set)

    def testRun_1(self):
        """test run monomer structure prediction"""
        self.monomer_objects_dir = self.test_data_dir
        self.oligomer_state_file = os.path.join(self.test_data_dir, "test_homooligomer_state.txt")
        self.args = [
            sys.executable,
            self.script_path,
            "--mode=homo-oligomer",
            "--num_cycle=1",
            "--num_predictions_per_model=1",
            f"--output_path={self.output_dir}",
            f"--data_dir={self.data_dir}",
            f"--oligomer_state_file={self.oligomer_state_file}",
            f"--monomer_objects_dir={self.monomer_objects_dir}",
            "--job_index=1"
        ]
        result = subprocess.run(self.args, capture_output=True, text=True)
        self._runCommonTests(result)

    def _runAfterRelaxTests(self, result):
        dirname = os.listdir(self.output_dir)[0]
        #Check if the directory contains five files starting from relaxed and ending with .pdb
        self.assertEqual(len([f for f in os.listdir(os.path.join(self.output_dir, dirname)) if f.startswith("relaxed") and f.endswith(".pdb")]), 5)

    def testRun_2(self):
        """test run without amber relaxation"""
        result = subprocess.run(self.args, capture_output=True, text=True)
        self._runCommonTests(result)
        #Check that directory does not contain relaxed pdb files
        dirname = os.listdir(self.output_dir)[0]
        self.assertEqual(len([f for f in os.listdir(os.path.join(self.output_dir, dirname)) if f.startswith("relaxed") and f.endswith(".pdb")]), 0)
        self.assertIn("Running model model_1_multimer_v3_pred_0", result.stdout + result.stderr)

    def testRun_3(self):
        """test run with relaxation for all models"""
        self.args.append("--models_to_relax=all")
        result = subprocess.run(self.args, capture_output=True, text=True)
        self._runCommonTests(result)
        self._runAfterRelaxTests(result)
        self.assertIn("Running model model_1_multimer_v3_pred_0", result.stdout + result.stderr)

    def testRun_4(self):
        """
        Test if the script can resume after all 5 models are finished, running amber relax on the 5 models
        """
        #Copy the example directory called "test" to the output directory
        shutil.copytree(os.path.join(self.test_data_dir,"P0DPR3_and_P0DPR3"), os.path.join(self.output_dir, "P0DPR3_and_P0DPR3"))
        self.args.append("--models_to_relax=all")
        result = subprocess.run(self.args, capture_output=True, text=True)
        self._runCommonTests(result)
        self._runAfterRelaxTests(result)
        self.assertIn("ranking_debug.json exists. Skipping prediction. Restoring unrelaxed predictions and ranked order", result.stdout + result.stderr)
        
    def testRun_5(self):
        """
        Test if the script can resume after 2 models are finished
        """
        # Copy the example directory called "test" to the output directory
        shutil.copytree(os.path.join(self.test_data_dir, "P0DPR3_and_P0DPR3_partial"), os.path.join(self.output_dir, "P0DPR3_and_P0DPR3"))
        result = subprocess.run(self.args, capture_output=True, text=True)
        self.assertIn("Found existing results, continuing from there", result.stdout + result.stderr)
        self.assertNotIn("Running model model_1_multimer_v3_pred_0", result.stdout + result.stderr)
        self.assertNotIn("Running model model_2_multimer_v3_pred_0", result.stdout + result.stderr)

        self._runCommonTests(result)

    def testRun_6(self):
        """
        Test running structure prediction with --multimeric_mode=True
        Checks that the output model follows provided template (RMSD < 3 A)
        """
        #checks that features contain pickle files
        self.assertTrue(os.path.exists(os.path.join(
            self.test_data_dir, "true_multimer", "features", "3L4Q_A.pkl")))
        self.assertTrue(os.path.exists(os.path.join(
            self.test_data_dir, "true_multimer", "features", "3L4Q_C.pkl")))
        self.args = [
            sys.executable,
            self.script_path,
            "--mode=custom",
            "--num_cycle=3",
            "--num_predictions_per_model=3",
            "--multimeric_mode=True",
            "--model_names=model_2_multimer_v3",
            "--msa_depth=128",
            f"--output_path={self.output_dir}",
            f"--data_dir={self.data_dir}",
            f"--protein_lists={self.test_data_dir}/true_multimer/custom.txt",
            f"--monomer_objects_dir={self.test_data_dir}/true_multimer/features",
            "--job_index=1"
        ]
        result = subprocess.run(self.args, capture_output=True, text=True)
        print(self.args)
        print(result.stdout)
        print(result.stderr)
        #self._runCommonTests(result) # fails because only one model is run
        from alphapulldown.analysis_pipeline.calculate_rmsd import calculate_rmsd
        rmsd_chain_b = []
        rmsd_chain_c = []
        reference = os.path.join(
            self.test_data_dir, "true_multimer", "modelling", "3L4Q_A_and_3L4Q_C", "ranked_0.pdb")
        for i in range(3):
            target = os.path.join(self.output_dir, "3L4Q_A_and_3L4Q_C", f"ranked_{i}.pdb")
            assert os.path.exists(target)
            rmsds = calculate_rmsd(reference, target)
            rmsd_chain_b.append(rmsds[0])
            rmsd_chain_c.append(rmsds[1])
            print(f"Model {i} RMSD chain B: {rmsds[0]}")
            print(f"Model {i} RMSD chain C: {rmsds[1]}")
        # Best RMSD must be below 2 A
        assert min(rmsd_chain_b) < 3.0
        assert min(rmsd_chain_c) < 3.0



#TODO: Add tests for other modeling examples subclassing the class above
#TODO: Add tests that assess that the modeling results are as expected from native AlphaFold2
#TODO: Add tests that assess that the ranking is correct
#TODO: Add tests for features with and without templates
#TODO: Add tests for the different modeling modes (pulldown, homo-oligomeric, all-against-all, custom)
#TODO: Add tests for monomeric modeling done

class TestFunctions(_TestBase):
    def setUp(self):
        #Call the setUp method of the parent class
        super().setUp()
        
        from alphapulldown.utils import create_model_runners_and_random_seed
        self.model_runners, random_seed = create_model_runners_and_random_seed(
            "multimer",
            3,
            1,
            self.data_dir,
            1,
        )

    def test_get_1(self):
        """Oligomer: Check that iptm+ptm are equal in json and result pkl"""
        self.output_dir = os.path.join(self.test_data_dir, "P0DPR3_and_P0DPR3")
        #Open ranking_debug.json from self.output_dir and load to results
        with open(os.path.join(self.output_dir, "ranking_debug.json"), "r") as f:
            results = json.load(f)
            #Get the expected score from the results
            expected_iptm_ptm = results["iptm+ptm"]["model_1_multimer_v3_pred_0"]
        
        pkl_path = os.path.join(self.test_data_dir, "P0DPR3_and_P0DPR3", "result_model_1_multimer_v3_pred_0.pkl")
        out = predict_structure.get_score_from_result_pkl(pkl_path)
        self.assertTupleEqual(out, ('iptm+ptm', expected_iptm_ptm))

    def test_get_2(self):
        """Oligomer: Check get_existing_model_info for all models finished"""
        self.output_dir = os.path.join(self.test_data_dir, "P0DPR3_and_P0DPR3")
        ranking_confidences, unrelaxed_proteins, unrelaxed_pdbs, START = predict_structure.get_existing_model_info(self.output_dir, self.model_runners)
        self.assertEqual(len(ranking_confidences), len(unrelaxed_proteins))
        self.assertEqual(len(ranking_confidences), len(unrelaxed_pdbs))
        self.assertEqual(len(ranking_confidences), len(self.model_runners))
        self.assertEqual(START, 5)
        with open(os.path.join(self.output_dir, "ranking_debug.json"), "r") as f:
            results = json.load(f)
            #Get the expected score from the results
            expected_iptm_ptm = results["iptm+ptm"]
        self.assertDictEqual(ranking_confidences, expected_iptm_ptm)

    def test_get_3(self):
        """Oligomer: Check get_existing_model_info, resume after 2 models finished"""
        self.output_dir = os.path.join(self.test_data_dir, "P0DPR3_and_P0DPR3_partial")
        ranking_confidences, unrelaxed_proteins, unrelaxed_pdbs, START = predict_structure.get_existing_model_info(self.output_dir, self.model_runners)
        self.assertEqual(len(ranking_confidences), len(unrelaxed_proteins))
        self.assertEqual(len(ranking_confidences), len(unrelaxed_pdbs))
        self.assertNotEqual(len(ranking_confidences), len(self.model_runners))
        self.assertEqual(START, 2)
        with open(os.path.join(self.output_dir, "ranking_debug_temp.json"), "r") as f:
            results = json.load(f)
            #Get the expected score from the results
            expected_iptm_ptm = results["iptm+ptm"]
        self.assertDictEqual(ranking_confidences, expected_iptm_ptm)

    #TODO: Test monomeric runs (where score is pLDDT)
    def test_get_4(self):
        """Monomer: Check that plddt are equal in json and result pkl"""
        pass

    def test_get_5(self):
        """Monomer: Check get_existing_model_info for all models finished"""
        pass

    def test_get_6(self):
        """Monomer: Check get_existing_model_info, resume after 2 models finished"""
        pass

if __name__ == '__main__':
    unittest.main()
