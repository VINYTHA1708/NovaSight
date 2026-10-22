# NovaSight Demo Example Images

These images are sample surfaces from the **NEU-CLS surface defect database** provided for application testing and demo evaluation on Hugging Face Spaces.

### Attribution
- **Dataset**: NEU Surface Defect Database (NEU-CLS)
- **Authors**: Kechen Song and Yunhui Yan (Northeastern University, China)
- **Citation**: K. Song and Y. Yan, *"A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects,"* Applied Surface Science, vol. 285, pp. 858–864, 2013.

### Image Inventory
1. **Known Defect Queries (from `sel_eval`)**:
   - `Cr_247.bmp`: Crazing defect (`Cr`)
   - `In_247.bmp`: Inclusion defect (`In`)
   - `Pa_247.bmp`: Patches defect (`Pa`)
2. **Held-out New Class Demonstration (Scratches `Sc`, from `held_enroll_pool`)**:
   - **5 Enrollment Examples**:
     - `Sc_189.bmp`
     - `Sc_148.bmp`
     - `Sc_101.bmp`
     - `Sc_13.bmp`
     - `Sc_29.bmp`
   - **1 Post-Enrollment Query**:
     - `Sc_192.bmp` (held-out query to test detection after enrollment)
