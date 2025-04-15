# DeepSDF 3D Shape Reconstruction

This project is my personal implementation of DeepSDF, focused on reconstructing 3D shapes using signed distance functions (SDFs). It supports training on `.obj` files, partial-view shape completion, and mesh generation via Marching Cubes.

---

### What It Does
- Trains a DeepSDF model on a set of 3D `.obj` files
- Samples 50,000 points per shape (surface + random)
- Learns per-shape latent embeddings
- Reconstructs meshes from predicted SDF values
- Supports shape completion using:
  - Nearest latent match
  - Latent code optimization

> *The DeepSDF paper uses 500,000 points per shape. I used 50,000 due to hardware limits, but more points generally yield better reconstructions. You can change this in the code by increasing the `num_samples` argument.*

---

### Chair Reconstructions (30 Total)

Trained over 10,000 epochs. Here are two sample outputs comparing the original and reconstructed mesh:

**Sample 0**  
![comparison_0](https://github.com/user-attachments/assets/52564b3e-50ab-43f2-b9cc-480f029d2e46)

**Sample 1**  
![comparison_1](https://github.com/user-attachments/assets/50b668ac-ad84-4610-a48c-650f214362df)

<details>
<summary>View Remaining 28 Reconstructions</summary>

![comparison_29](https://github.com/user-attachments/assets/f8dd1107-0732-4a97-b777-19213e1ca927)
![comparison_28](https://github.com/user-attachments/assets/adfae8fa-217e-47bb-8f75-48565b0b1994)
![comparison_27](https://github.com/user-attachments/assets/20334f97-bf76-4c7c-a38a-30b899369ebf)
![comparison_26](https://github.com/user-attachments/assets/22a8dc81-3282-45ff-abda-ace56e213649)
![comparison_25](https://github.com/user-attachments/assets/a70ea4d0-84ae-40bc-9472-a2a9a686da9a)
![comparison_24](https://github.com/user-attachments/assets/2d5dd91a-0938-4ec4-9fe9-ce400d6ace3a)
![comparison_23](https://github.com/user-attachments/assets/a6b4f696-dddf-47d4-ac05-875d52b9409d)
![comparison_22](https://github.com/user-attachments/assets/38b2ec85-b154-4657-8cce-ae2a8cdfef5c)
![comparison_21](https://github.com/user-attachments/assets/2cbd3595-c787-4059-89a9-0b60eaab0644)
![comparison_20](https://github.com/user-attachments/assets/6976a67d-cb0e-4ae9-9206-a39f03290155)
![comparison_19](https://github.com/user-attachments/assets/ac9aec58-6e47-44b6-ba7a-53bd0fc663ec)
![comparison_18](https://github.com/user-attachments/assets/73455ca2-6a84-4908-9db9-e9521db430bc)
![comparison_17](https://github.com/user-attachments/assets/739dfded-a7d2-400f-b3c9-175af33f6196)
![comparison_16](https://github.com/user-attachments/assets/3455e723-39cd-4adf-a974-6aa2e7d2bdd8)
![comparison_15](https://github.com/user-attachments/assets/94de11af-759a-4f0e-a743-5e51fc14c302)
![comparison_14](https://github.com/user-attachments/assets/c7b022a6-9a12-4d36-8952-8a3ffc78322d)
![comparison_13](https://github.com/user-attachments/assets/9436e5ee-b471-4d0c-84e9-85ec3a337251)
![comparison_12](https://github.com/user-attachments/assets/c6c71d9a-322d-46ed-9411-a5f2e12db4be)
![comparison_11](https://github.com/user-attachments/assets/f263184b-0406-4341-8d01-5307f0abbbda)
![comparison_10](https://github.com/user-attachments/assets/1bd68dba-2cf4-4482-b1d4-d1b187ab9d01)
![comparison_9](https://github.com/user-attachments/assets/3b2f375d-0151-4b94-88f8-6a7164223a75)
![comparison_8](https://github.com/user-attachments/assets/8b2dd4ce-b58e-4c2c-9af7-d4024e0c20c5)
![comparison_7](https://github.com/user-attachments/assets/99a1386b-6394-482e-b232-13cfa93c195d)
![comparison_6](https://github.com/user-attachments/assets/09cf54fe-edbe-4cf1-8b77-594b96ba8cc3)
![comparison_5](https://github.com/user-attachments/assets/87805399-7a59-4008-9aa8-4c4eda51ab96)
![comparison_4](https://github.com/user-attachments/assets/9950b893-ca2f-458f-94a5-a10347492633)
![comparison_3](https://github.com/user-attachments/assets/9a9d9421-30cd-4d17-abee-b144e0024353)
![comparison_2](https://github.com/user-attachments/assets/a1a9d12c-8d6d-4894-8bd6-798b6cae21a7)

</details>

---

### Shape Completion

Given only a partial input (e.g. front view), the model can:

**Best Match from Latent Bank**  
![comparison_best_match](https://github.com/user-attachments/assets/980986c6-75e2-4147-ad76-d28ef6ef2627)

**Latent Optimization**  
![comparison_optimized](https://github.com/user-attachments/assets/98e9ec12-7ae6-49a4-ae28-fa9789046a2e)

---

### Bunny Reconstruction

This bunny was trained using 50,000 sampled points over 5,000 epochs. Since it’s a simple object compared to a chair, the result came out clean and accurate.

**Original**  
![image](https://github.com/user-attachments/assets/beef3ab7-99d5-4b72-bc4b-3b93e8578578)

**Reconstructed**  
![image](https://github.com/user-attachments/assets/48f43ac8-1459-442b-b8f4-000991994c7e)

---

### Experiments in Progress

#### Text-Guided Latent Editing
Testing how an LLM can modify latent codes from prompts like _“a chair with curved legs.”_. The LLM interprets phrases like “a chair with curved legs” and adjusts latent space dimensions accordingly.  

![Prompt Editing](https://github.com/user-attachments/assets/a64dbece-eb0b-44f1-b8eb-1059922b8ece)

#### Latent Space Visualization
Plotting shape embeddings to explore clustering, interpolation, and prompt-based transformations.

![Latent Space Map](https://github.com/user-attachments/assets/6de205dc-aa29-4d00-8598-b810a791495a)

These tools help link semantic meaning and 3D geometry. Still testing accuracy and generation quality but early results are interesting.

---

### Code & Usage Notes

This code was originally written just for me and is not yet modular or generalized. I’m planning to clean it up and release a version that works out-of-the-box on any dataset.

If you're interested in trying it now:
- You’ll need to manually update the paths for `obj_path`, `save_dir`, etc.
- It works best on ShapeNet Core `.obj` meshes
- `NOT_RECONSTRUCTED = True` toggles training vs test mode

#### Dependencies
```bash
pip install torch trimesh scikit-image matplotlib numpy
```

#### Run
```bash
python DeepSDFCode.py
```

---

### What’s Next

- Clean up and modularize the codebase
- Add CLI for custom datasets and checkpoint control
- Finish LLM integration and latent-guided generation
- Publish code once more stable

If you have questions or want help running this, feel free to reach out. I’ll be happy to explain anything.

---

### Paper Reference

[DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.pdf)

---

### Bunny Latent Code

Example of a trained latent vector used to reconstruct the Stanford Bunny.

```txt
{2.571929804980754852e-02
-1.891424879431724548e-02
-2.272037602961063385e-02
-3.282329766079783440e-03
2.671889029443264008e-02
-1.568010449409484863e-02
-1.949955709278583527e-02
1.988929882645606995e-02
-2.002627216279506683e-02
-5.950008518993854523e-03
2.137670479714870453e-02
1.250777766108512878e-02
2.193937823176383972e-02
-1.024499349296092987e-02
6.561588961631059647e-03
1.607674919068813324e-02
-2.690345235168933868e-02
1.764042116701602936e-02
1.839219965040683746e-02
2.136653102934360504e-02
2.471710927784442902e-02
1.862458698451519012e-02
8.582584559917449951e-03
2.496489696204662323e-02
-4.940207581967115402e-03
1.576032303273677826e-02
-1.771047100191935897e-04
1.025762408971786499e-02
-2.179501205682754517e-02
-1.298787910491228104e-02
2.090901695191860199e-02
-1.437296066433191299e-02
2.377056516706943512e-02
4.118666984140872955e-03
-2.042565681040287018e-02
2.172596752643585205e-02
2.784682251513004303e-02
2.780549041926860809e-02
-1.262331102043390274e-02
2.527266740798950195e-02
2.769804000854492188e-02
-7.739173714071512222e-03
-2.074931748211383820e-02
-7.500117644667625427e-03
1.348469965159893036e-02
-1.210085395723581314e-02
-2.020902000367641449e-02
-8.486658334732055664e-03
-1.604524441063404083e-02
-2.227163501083850861e-02
-3.943101502954959869e-03
-2.004387788474559784e-02
-1.141452789306640625e-02
1.389472465962171555e-02
1.287480466999113560e-03
-1.063225232064723969e-02
-1.090580690652132034e-02
-1.872522383928298950e-02
-1.800170727074146271e-02
-6.147798616439104080e-03
-1.314948219805955887e-02
-1.917962916195392609e-02
-4.188995808362960815e-03
-1.953523047268390656e-02
1.575730927288532257e-02
1.653507165610790253e-02
-1.710680127143859863e-02
1.698636449873447418e-02
7.092911284416913986e-03
3.082667663693428040e-02
1.420361734926700592e-02
2.613485325127840042e-03
2.594907954335212708e-02
2.252011001110076904e-02
1.041685137897729874e-02
2.781263366341590881e-02
2.060546353459358215e-02
-1.181704830378293991e-02
-6.185079459100961685e-03
-2.402738109230995178e-02
1.151771191507577896e-02
-2.488913759589195251e-02
-1.272862404584884644e-02
-2.601240994408726692e-03
6.778646260499954224e-03
-2.375466376543045044e-02
1.616838946938514709e-02
-9.508697316050529480e-03
1.516710966825485229e-02
-2.343936264514923096e-02
-2.233266830444335938e-02
2.641224674880504608e-02
-1.049401331692934036e-02
1.826234348118305206e-02
2.033309265971183777e-02
1.292957924306392670e-02
-2.118060551583766937e-02
-3.032799577340483665e-03
1.503079291433095932e-02
-1.850883662700653076e-02
1.203035190701484680e-02
2.134085074067115784e-02
2.147503569722175598e-02
9.650061838328838348e-03
-1.575212366878986359e-02
-1.155801489949226379e-02
1.429485809057950974e-02
-2.237617783248424530e-02
-1.444547809660434723e-02
-1.785629987716674805e-02
-2.891577221453189850e-02
2.882771193981170654e-02
2.162054367363452911e-02
-2.001342736184597015e-02
-2.540876902639865875e-02
-1.157671213150024414e-02
-2.842639572918415070e-02
-1.069029141217470169e-02
1.705634035170078278e-02
2.369702793657779694e-02
8.442553691565990448e-03
1.473719719797372818e-02
3.313469886779785156e-02
1.589295081794261932e-02
2.056014165282249451e-02
-2.329988777637481689e-02
2.158742398023605347e-02
-2.093892544507980347e-02
}
```





