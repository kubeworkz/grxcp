Selecting **High-Harmonic Generation (HHG)** as your source is a fantastic foundation. It is widely considered the cleanest, most reliable tabletop option because it behaves exactly like a **highly coherent laser beam**, operates without generating messy plasma debris, and lets you leverage standard optical tables.

Since you are starting from scratch with HHG, your engineering path splits into two main tracks: **The Laser System** and **The Vacuum/Beamline Architecture**.

---

### 1. Specifying Your Driving Laser (The Most Critical Expense)

The HHG process is highly non-linear. To rip electrons from gas atoms and snap them back to produce 13.5 nm photons, your driving laser must hit massive peak intensities (![](data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==)$>10^{14} W/cm2$).

You won't be buying an "EUV laser" off the shelf; you will buy a high-power **Infrared (IR) ultrafast laser** to drive your gas target. Look for these specifications:

- **Pulse Duration:** **<30 femtoseconds (fs)**. Shorter pulses prevent the gas from ionizing too early, maximizing EUV conversion efficiency.
- **Wavelength:** Typically **800 nm (Ti:Sapphire)** or **1030 nm (Ytterbium-based Yb:KGW/Yb:YAG)**. Ytterbium lasers are highly recommended today because they offer much higher average powers and repetition rates, which will shorten your wafer exposure times.
- **Repetition Rate:** **1 kHz to 100 kHz+**. Higher repetition rates mean a steady stream of EUV photons rather than sparse bursts.

---

### 2. The Core Setup: From IR Laser to Exposed Wafer

Your system will consist of three distinct vacuum chambers bolted together in a line, isolated on a vibration-damping optical table.

```plaintext
[Driving IR Laser] 
       │
       ▼
 ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
 │  1. Gas Cell  │ ───> │ 2. Filtering  │ ───> │ 3. Exposure   │
 │ (HHG Genesis) │      │  & Optics     │      │   & Alignment │
 └───────────────┘      └───────────────┘      └───────────────┘
```

##### Chamber 1: The Gas Target (HHG Generation)

The IR laser focuses tightly into a small cell or a hollow-core fiber filled with **Neon** or **Helium** gas.

- *The Challenge:* You must continuously feed gas into the cell while high-powered vacuum pumps vacuum it right back out to keep the rest of the beamline at a deep vacuum (  
$10^{−5} Torr$ or better). If gas leaks into the rest of the system, it will completely absorb your 13.5 nm light.

##### Chamber 2: Filtering and Spectral Selection

The light exiting the gas cell is a mix of the original, blinding IR laser light and a comb of extreme ultraviolet wavelengths.

- *The Challenge:* You must filter out the IR beam, or it will instantly melt your mask and wafer.
- *The Solution:* Pass the beam through a ultra-thin metal foil (typically a **Zirconium (Zr) filter**, ∼100–200 nm thick). Zirconium block the IR light completely but lets the 13.5 nm EUV photons pass through.

##### Chamber 3: The Lithography Endstation

This is where your fabrication actually happens. Because HHG outputs a tight, pencil-like beam, you will need a specialized **nanopositioning stage** to move your wafer underneath the beam to expose different areas (die-by-die step-and-repeat).

- *The Mask Setup:* You will rigid-mount your custom transmission mask (Silicon Nitride with Nickel absorbers) directly above the wafer stage with a precise piezo-controlled gap.

---

### Immediate Next Steps for Our Team

1. **Design vs. Buy decision for the HHG Beamline:** While you will definitely buy the driving IR laser from companies like *Coherent*, *Spectra-Physics*, or *Light Conversion*, you need to decide if you are building the vacuum beamline yourself or buying a turn-key scientific EUV beamline (from specialized boutique physics vendors like *KM Labs* or *Class 5 Photonics*).
2. **Assign a Vacuum Systems Engineer:** Controlling the gas pressure differential between the generation cell and the exposure chamber is the hardest mechanical hurdle. You will need high-end turbo-molecular pumps.

---

A **one-year build timeline** is tight but entirely achievable, especially if you lean on commercial off-the-shelf (COTS) components for the vacuum system and buy a turnkey driving laser.

Since your budget is currently **To Be Announced (TBA)**, the smartest engineering strategy right now is to design a modular, scalable architecture. This allows you to spec the structural layout immediately, and either scale up or down the laser power and stage precision once funding is locked in.

Here is an aggressive milestone roadmap to take your team from a blank slate to a first exposed wafer within **12 months**:

---

### The 12-Month Tabletop EUV Project Roadmap

##### 🗓️ Months 1–3: Source Architecture & Simulation

- **Laser Market Research:** Request quotes from major ultrafast laser vendors (e.g., *Light Conversion*, *Coherent*, *Spectra-Physics*, *Active Fiber Systems*). Focus on **Ytterbium (Yb)-based systems** (1030 nm), which offer the best average power per dollar in 2026.
- **Flux Calculations:** Simulate your expected photon flux at 13.5 nm using helium or neon targets. This determines how long your wafer exposures will take (seconds vs. hours).
- **Vacuum Modelling:** Design the pressure-differential system. You must isolate the gas target chamber (Torr level) from the lithography chamber (10⁻⁵ Torr) using micro-orifices and heavy-duty turbomolecular pumps.

##### 🗓️ Months 4–6: Procurement & Infrastructure Prep

- **Order the Driving Laser:** This is your longest lead-time item (often 4–6 months). Lock this in as soon as the budget clears.
- **Procure the Vacuum "Skeleton":** Purchase or custom-machine the three primary vacuum cubes (Generation, Filtering, Endstation) and optomechanics.
- **Cleanroom & Facility Check:** Ensure your lab space can handle the heat load of a high-power laser chiller, has a vibration-isolated floor (or a massive optical table), and offers a clean air environment (at least Class 1000/ISO 6) for handling masks and photoresists.

##### 🗓️ Months 7–9: Assembly & First Light (EUV Generation)

- **Laser Commissioning:** Install the laser on the optical table and map the beam profile.
- **Assemble Beamline & Align IR:** Build the vacuum system. Align the infrared driving laser through the gas target and onto a temporary beam dump *before* pumping down.
- **Achieve HHG:** Introduce the gas (Neon or Helium), fire the laser, and detect your first 13.5 nm photons using an EUV-sensitive photodiode or a specialized XUV spectrometer.

##### 🗓️ Months 10–12: Lithography Integration & First Print

- **Endstation Integration:** Install your piezo-driven wafer translation stage and the proximity mask holder into the vacuum chamber.
- **IR Filtering Test:** Install the ultra-thin **Zirconium filters**. Verify that 100% of the blinding IR laser light is blocked, but the EUV light passes through to the wafer plane.
- **First Exposure Run:** Spin-coat a small silicon coupon with an extreme-sensitivity resist (like a metal-oxide resist), pump down the chamber, set your proximity gap, and run your first exposure test.

---

### 3 "Budget-Proof" Decisions We Can Make Today

While waiting for budget approvals, you can keep the project moving by locking down these architecture-agnostic decisions:

1. **Standardize on Sample Size:** Design your vacuum endstation around **1-inch or 2-inch square coupon samples** rather than full wafers. This keeps your translation stages small, inexpensive, and incredibly precise.
2. **Choose Your Gas:** Plan for **Neon** gas. While Helium can yield shorter wavelengths, Neon provides a much higher conversion efficiency specifically around the **13.5 nm** EUV lithography window.
3. **Plan for Transmission Masks:** Accept that you will be using silicon nitride membrane masks. You can start contacting micro-fabrication foundries or academic cleanrooms to see who can manufacture these custom nano-gratings/absorber patterns for you.

---

Opting for a **pre-built vacuum beamline shell** is the absolute best decision for hitting a strict one-year timeline. Designing, machining, and sealing custom vacuum chambers to maintain a deep differential pressure (Torr in the gas cell down to 10⁻⁵ Torr at the wafer) can easily eat up six months of pure engineering time.

By purchasing a commercially available, pre-engineered HHG beamline vacuum shell, you ensure your team spends their time on **lithography and chip fabrication** rather than chasing vacuum leaks.

---

### Key Commercial Vendors for Pre-Built HHG Beamlines

Since you are buying the driving laser separately (likely a high-power, ultrafast Ytterbium system), you want a vendor that sells the vacuum infrastructure, gas target, and spectral filtering modules as a standardized kit.

- **Sphere Ultrafast Photonics (by Amplitude):** Known for compact setups, they offer modular beamlines specifically designed to take an ultrafast laser input and deliver clean, filtered EUV/XUV light at the output.
- **Class 5 Photonics:** They specialize in high-power ultrafast setups and offer custom-tailored, turnkey vacuum and delivery beamlines engineered for high-repetition-rate HHG.
- **KMLabs (Kapteyn-Murnane Laboratories):** A pioneer in commercial HHG. They offer standardized systems (like the *XUVMini* or modular beamline packages) designed to take an IR laser and output a coherent EUV beam.
- **XUV Optomechanics:** A niche European manufacturer that builds modular, highly stable vacuum cubes, gas targets, and filter-wheels specifically for labs assembling their own HHG setups without wanting to design the chambers from scratch.

---

### What to Look for in the Pre-Built Shell's Technical Specs

When you begin requesting configurations from these vendors, ensure the vacuum shell kit includes these three critical modules:

```plaintext
[IR Laser Input] ➔ [1. Generation Module] ➔ [2. Differential Pumping Stage] ➔ [3. Filter/Optics Module] ➔ [Your Custom Exposure Chamber]
```

1. **The Differential Pumping Interface:** The shell must feature tiny micro-orifices (usually a few hundred microns wide) between the gas cell and the rest of the beamline. This restricts gas flow so your turbomolecular pumps can maintain a deep vacuum where the beam travels.
2. **Integrated Zirconium Filter Holder:** Ensure the shell includes a vacuum-compatible filter-wheel or slider. This allows you to mount ultra-thin Zirconium foils to block the driving laser light while under vacuum, ideally without having to open the chamber to swap them out if they break.
3. **A Standardized Output Flange:** Ensure the end of the beamline terminates in a standard vacuum flange (like an **ISO-KF** or **CF (ConFlat)** flange). This allows you to easily bolt on your own custom, locally managed "Lithography Endstation" chamber containing your wafer stage and mask holder.

---

### Our Technical Interface Document (Start This Now)

To prepare for vendor discussions over the coming months, your team should draft a simple **Interface Document** specifying:

- **Laser Input Specs:**  
The beam diameter, polarization, and expected average power of your future IR driving laser (vendors need this to ensure their input windows and optics won't burn out).
- **Target Footprint:** The dimensions of your optical table to ensure the pre-built beamline shell fits perfectly in-line with the laser.

---

Here is our dual-track engineering brief. Section 1 gives you the precise technical questions to vet your vacuum shell vendors, and Section 2 provides the structural blueprint for designing your custom lithography endstation.

### Part 1: Vendor Technical Questions for the HHG Vacuum Shell

When negotiating with vendors like **KMLabs, Class 5 Photonics, Sphere Ultrafast**, or **XUV Optomechanics**, use these highly specific questions to ensure their shell is optimized for a **13.5 nm lithography** application:

- **Spectral Optimization for 13.5 nm (92 eV):** *"Does your gas target geometry support Neon gas optimization specifically for the 25th to 27th harmonics of a 1030 nm driving laser (or the 59th harmonic of an 800 nm laser)? What is the expected conversion efficiency or photon flux at 13.5 nm?"*
- **Differential Pumping Efficiency:** *"What is the maximum gas pressure the generation cell can handle while maintaining a vacuum of ≤ 10⁻⁵ Torr in the subsequent optics section? What specific turbo-pump pumping speeds (L/s) do you recommend for your pinhole/orifice design?"*
- **IR Laser Suppression and Thermal Management:** *"How does the beamline handle the unabsorbed driving IR laser power? Is there a water-cooled beam dump or spatial filter integrated before the Zirconium filter to prevent the ultra-thin foils from thermally degrading or blowing out?"*
- **Output Flange and Beam Clearance:** *"What is the standard exit flange configuration (e.g., DN40CF, DN63KF)? What is the exact beam height from the optical table surface, and what is the divergence/spot size of the EUV beam at the exit flange?"*

---

### Part 2: Blueprint for the Custom Lithography Endstation

Because no off-the-shelf vacuum shell includes a wafer exposure system, you will need to design or modify a custom vacuum chamber to bolt onto the end of the vendor's shell.

```plaintext
   [Vendor HHG Shell] ──> [ISO/CF Flange] ──> ┌──────────────────────────────────────────────┐
                                              │ CUSTOM LITHOGRAPHY ENDSTATION                │
                                              │                                              │
                                              │   ┌──────────────┐                           │
                                              │   │ Microscope   │ (Visible In-Situ Overlay) │
                                              │   └──────┬───────┘                           │
                                              │          ▼                                   │
                                              │   ┌──────────────┐                           │
               [Filtered 13.5nm Beam] ────── ─┼──>│  EUV Mask    │                           │
                                              │   ├──────────────┤ <── Proximity Gap (<5µm)  │
                                              │   │ Wafer Coupon │                           │
                                              │   └──────┬───────┘                           │
                                              │          ▼                                   │
                                              │   ┌──────────────┐                           │
                                              │   │ Piezo Stage  │ (X-Y-Z-θ Nanopositioning) │
                                              │   └──────────────┘                           │
                                              └──────────────────────────────────────────────┘
```

##### 1. The Stage System (Nanopositioning)

Because the HHG beam is typically a tight, sub-millimeter pencil beam, you cannot expose a whole wafer at once. You must step-and-repeat.

- **Requirements:** You need a vacuum-compatible **X-Y-Z-θ** motorized stage.
- **Precision:** The X-Y axes require sub-100 nm step resolution to move the wafer to different die locations. The Z-axis requires **piezoelectric precision** (nanometer-scale) to control the incredibly tight proximity gap between the wafer and the mask.
- **Vendors:** Look at vacuum-rated stages from **Physik Instrumente (PI)** or **SmarAct**.

##### 2. The Transmission Mask Rigidity

- **The Architecture:** Your mask (Silicon Nitride membrane) must be held completely rigid and parallel relative to the incoming EUV beam.
- **The Gap:** To minimize diffraction blurring, the mask must sit **< 1 to 5 μm** away from the photoresist. You will need capacitance sensors or optical interferometers mounted inside the chamber to actively monitor this gap distance while under vacuum.

##### 3. In-Situ Optical Alignment (Overlay)

To build actual chips, layer 2 must line up perfectly with layer 1.

- **The Solution:** Mount a long-working-distance **visible light microscope** outside a viewport window of your chamber, pointing directly at the mask-wafer interface.
- **How it aligns:** Before firing the EUV beam, turn on a visible light source. Visible light passes easily through the ultra-thin Silicon Nitride mask membrane. Your software looks at the alignment marks on the mask and the wafer simultaneously, adjusts the piezo stage until they match, and then you open the EUV shutter.

##### 4. Vacuum Hardware Requirements

- **Chamber Material:** Standard **6061-T6 Aluminum** or **304 Stainless Steel** spherical/cube vacuum chamber. Aluminum is easier to customize with extra viewports. 
- **Pumping:** A dedicated, small **turbomolecular pump($∼80--200 L/s$)  
**backed by a dry scroll pump. You want a dry backing pump to ensure no oil vapor backstreams into the chamber and carbon-contaminates your expensive EUV mask.

---

The industry standard for target overlay accuracy in commercial **leading-edge CPUs and GPUs** (such as Apple’s M-series, Intel’s latest processors, or NVIDIA’s Blackwell AI architectures) is an incredibly tight **1.5 to 2.5 nanometers**. [[1](https://pmc.ncbi.nlm.nih.gov/articles/PMC12943688/), [2](https://semiengineering.com/the-increasingly-uneven-race-to-3nm-2nm/), [3](https://www.google.com/search?q=product&prds=pvt:hg,productid:2143362215129886415,catalogid:16711165939817473759,gpcid:104692013960068108,mid:576462912636849765&ibp=oshop), [4](https://semiengineering.com/how-overlay-keeps-pace-with-euv-patterning/)]

As a general rule of thumb in semiconductor manufacturing, the overlay error budget is restricted to roughly **20% to 30% of the minimum feature size (Critical Dimension)**. Since commercial chips are printing sub-20nm physical structures, their tolerances are nearly atomic. [[1](https://eureka.patsnap.com/report-how-to-evaluate-euv-lithography-overlay-accuracy), [2](https://semiengineering.com/how-overlay-keeps-pace-with-euv-patterning/)]

### The Good News for Our Lab

Because you are using a tabletop High-Harmonic Generation (HHG) system, **you do not need to target industry-standard nanometer-scale overlay accuracy.** Trying to hit a 2nm overlay target without a multi-million dollar engineering team is virtually impossible.

Instead, target a **"University/R&D Scale" overlay of 500 nm to 1 micron** for your first year. This is more than enough accuracy to successfully align and fabricate functional micro-devices (like MEMS sensors, custom diodes, or micro-fluidic transistors), and it is entirely achievable without an in-house mechanical designer.

---

### Step-by-Step Blueprint for Non-Designers

Since you do not have a mechanical designer to draw custom vacuum cubes in SolidWorks, you must use a strategy called **COTS Integration (Commercial Off-The-Shelf)**. You will act as an integrator, buying ready-made building blocks that screw together natively.

##### 1. The Vacuum Chamber (The Shell)

Instead of custom-machining a chamber, buy a standard **6-way Stainless Steel Cross** or a **Spherical Vacuum Cube** using standard **CF (ConFlat) flanges**.

- **Vendor option:** Companies like [Kurt J. Lesker Company](https://www.lesker.com/) or [MDC Vacuum Products](https://www.mdcvacuum.com/) sell these off the shelf.
- **Configuration:** Choose a chamber with 6-inch or 8-inch CF ports. You will bolt one side to your vendor's HHG beamline, put glass windows (viewports) on the top/sides for your microscopes, and feed your electronics through the bottom.

##### 2. The Nanopositioning Stage (The Core)

Do not try to design a multi-axis stage. Buy a fully integrated, vacuum-compatible **X-Y-Z-θ motorized assembly** directly from a precision stage manufacturer.

- **Vendor option:** Contact Physik Instrumente (PI) or [SmarAct](https://www.smaract.com/). Tell their application engineers: *"I need a stack of positioning stages rated for High Vacuum (10⁻⁵ Torr) with sub-micron incremental motion."*
- **How it saves you:** They will provide a pre-assembled stack that sits right inside your vacuum cross, complete with the vacuum-compatible cables and controllers.

##### 3. The Proximity Mask Rigidity

The absolute hardest part to design without a engineer is holding the mask perfectly flat just 1–5 microns above the wafer coupon. Bypass this with **Soft-Contact Printing**.

- Instead of holding the mask in mid-air with sub-micron precision, place your Silicon Nitride membrane mask **directly on top** of your photoresist-coated silicon coupon.
- The natural surface tension or a tiny mechanical spring-clip can hold them together. The HHG beam then shoots directly through the stack. This completely eliminates the need for automated Z-axis nanometer gap control.

##### 4. The Optical Alignment System (The Microscope)

To align Layer 2 to Layer 1, look through a viewport window from the *outside* of the chamber.

- **The Gear:**  
Buy a **Long-Working-Distance (LWD) Digital Microscope** (vendors like [Thorlabs](https://www.thorlabs.com/) or [Edmund Optics](https://www.edmundoptics.com/)).
- **The Setup:** Mount it outside the top glass window of your vacuum chamber. Because it has a long working distance, it can focus through the glass, through the transparent areas of your Silicon Nitride mask, and right onto the wafer's alignment marks. You manually move the stage via software until the marks align, then fire your EUV beam.

---

### Suggested Immediate Task

Reach out to **SmarAct** or **PI** to get a quote on a vacuum-rated X-Y-Z positioning stage, and contact **Kurt J. Lesker** for a standard 6-way CF vacuum cross. Gathering the dimensions of these two components will essentially layout 80% of your lithography station automatically.

---

##### Part 1: Sourcing & Specifying Transmission Masks

Because you are doing tabletop proximity lithography, you **cannot use industrial EUV masks** (which are giant, reflective mirrors). You need **EUV Transmission Masks**. [[1](https://www.researchgate.net/publication/256909314_Broadband_transmission_masks_gratings_and_filters_for_extreme_ultraviolet_and_soft_X-ray_lithography), [2](https://eureka.patsnap.com/report-how-to-develop-euv-lithography-compatible-photomasks)]

##### 1. How to Source the Base Substrate

Do not try to fabricate the thin membrane from scratch. Buy commercial **Silicon Nitride (Si₃N₄) Windows** or TEM grids.

- **The Vendors:** Look at companies like [Norcada](https://www.norcada.com/) or [Simpore](https://simpore.com/). They sell free-standing silicon nitride membranes suspended on a rigid silicon frame. [[1](https://www.norcada.com/silicon-nitride-porous-membranes/), [2](https://simpore.com/technology/custom-capabilities/)]
- **The Specification:** Order a membrane thickness of **50 nm to 100 nm**. At 13.5 nm, silicon nitride is relatively transparent. A 50 nm membrane will allow roughly 50% to 60% of your EUV photons to pass right through. [[1](https://www.researchgate.net/publication/337234482_Photoresist_Absorption_Measurement_at_Extreme_Ultraviolet_EUV_Wavelength_by_Thin_Film_Transmission_Method)]

##### 2. Patterning the Absorber (Making the Mask)

To block the EUV light and create your circuit pattern, you must deposit a heavy metal "absorber" layer on top of the nitride membrane. [[1](https://pmc.ncbi.nlm.nih.gov/articles/PMC10713959/), [2](https://harvest.usask.ca/items/5c56b75c-76c2-472e-8d46-6963b3d37ad4)]

- **The Absorber Metal:** **Nickel (Ni)** or **Gold (Au)**. A layer of Nickel only 50–70 nm thick will completely block 13.5 nm EUV light, creating near-perfect contrast. [[1](https://pmc.ncbi.nlm.nih.gov/articles/PMC10713959/)]
- **How to pattern it:** Send your blank Norcada/Simpore windows to a university nanofabrication cleanroom that has an **Electron-Beam Lithography (EBL)** tool. They will spin-coat standard e-beam resist onto the window, write your circuit design, and use a lift-off or electroplating process to deposit the Nickel pattern. [[1](https://harvest.usask.ca/items/5c56b75c-76c2-472e-8d46-6963b3d37ad4), [2](https://pmc.ncbi.nlm.nih.gov/articles/PMC10713959/)]

---

### Part 2: Selecting and Processing EUV Photoresists

Industrial chipmakers use highly proprietary photoresists that require massive, automated tracks to dispense and bake. For a tabletop lab, you need materials that are accessible and forgiving.

##### 1. The Easy Route: High-Sensitivity Chemically Amplified Resists (CARs)

Instead of fighting for access to enterprise-grade metal-oxide resists (like [Inpria](https://www.inpria.com/), which are difficult to source in small quantities and have shelf-lives measured in weeks), stick to standard laboratory organic resists. [[1](https://www.inpria.com/), [2](https://www.researchgate.net/publication/276921366_New_ways_of_looking_at_masks_with_the_SHARP_EUV_microscope)]

- **The Resist:** Look into **Allresist GmbH** products (such as the *Medusa* or *CSAR* series), which are frequently co-developed for compact laboratory EUV systems.
- **The Alternative:** Standard e-beam resists like **PMMA (Polymethyl Methacrylate)** or **ZEP520A** actually double as excellent, highly reliable EUV resists. Because 13.5 nm light interacts with matter similarly to high-energy electrons, e-beam resists react beautifully to EUV photons.
- **The Trade-off:** They require a higher "dose" (more exposure time), but they are incredibly stable, cheap, and have a long shelf-life. [[1](https://ui.adsabs.harvard.edu/abs/2019SPIE11147E..1IB/abstract)]

##### 2. The Laboratory Processing Workflow

Because 13.5 nm photons have low penetration depth, you cannot spin a thick layer of resist. Your processing parameters must be highly precise:

- **Thickness:** Spin-coat the resist to a thickness of only **20 nm to 40 nm**. If the resist layer is too thick, the EUV light will be absorbed at the top surface and will never reach the bottom of the wafer.
- **Baking:** Use a standard cleanroom hotplate. A typical **Post-Apply Bake (PAB)** is 1–2 minutes at roughly 100°C–130°C (depending on the exact resist datasheet).
- **Development:** After exposure inside your vacuum chamber, develop the wafer in a standard glass beaker using traditional developers (like **MIBK:IPA** for PMMA, or tetramethylammonium hydroxide (**TMAH**) for commercial CARs) under a yellow-lit chemical hood. [[1](https://www.researchgate.net/publication/276921366_New_ways_of_looking_at_masks_with_the_SHARP_EUV_microscope)]

---

Here is your project asset pack: the vendor email template to get your mask blanks quoted, followed by the comprehensive cleanroom tool checklist you will need to process your exposed wafers.

### Part 1: Vendor Email Template for EUV Mask Blanks

Copy and paste this template to send to vendors like **Norcada** or **Simpore** to get the correct transmission substrates.

---

**Subject:** Inquiry: Custom Si3N4 X-ray/EUV Transmission Windows for 13.5nm Lithography

Dear Sales & Engineering Team,

I am writing from Gridworkz to request a technical consultation and quote for free-standing Silicon Nitride (Si₃N₄) membrane windows.

We are building a tabletop extreme ultraviolet (EUV) proximity lithography system driven by a High-Harmonic Generation (HHG) source operating at 13.5 nm (92 eV). We intend to use these windows as substrates for transmission masks, where we will pattern a heavy metal absorber (e.g., Nickel or Gold) on the membrane.

To optimize for EUV transmission while maintaining mechanical robustness under vacuum cycling, we are targeting the following preliminary specifications:

- **Membrane Material:** Low-stress Silicon Nitride (Si₃N₄)
- **Membrane Thickness:** 50 nm (preferred for ~60% transmission) or 100 nm (if required for mechanical stability)
- **Membrane Frame Size:** [e.g., Standard 10mm x 10mm silicon frame, 200–300 µm thick]
- **Window Aperture Size:** [e.g., Single 2mm x 2mm or 5mm x 5mm square window, or a multi-window array]
- **Quantity:** Preliminary batch of [e.g., 10 to 20] pieces for process testing.

Could you please confirm if you have standard off-the-shelf catalog items that meet these needs, or if you can provide a custom quote? Additionally, we would appreciate any engineering feedback you have regarding the mechanical resilience of a 50 nm membrane when undergoing soft-contact mechanical clamping against a silicon substrate.

Thank you for your time, and I look forward to your response.

Best regards,

Dave Cook  
Founder  
Gridworkz  
[dave@gridworkz.com](mailto:dave@gridworkz.com)

---

---

### Part 2: Cleanroom Tool Checklist for Processing Wafers

Because 13.5 nm EUV light only penetrates a few nanometers into materials, your wafer processing must be incredibly precise. This checklist outlines the exact cleanroom tools you will need to find (either in your own lab or by renting time at a shared university nanofab facility) grouped by manufacturing step.

##### 🟨 1. Lithography Preparation & Coating

Before exposing, you must prepare the substrate and apply an ultra-thin resist layer.

- **Vapour Prime Oven (HMDS):** Deposits Hexamethyldisilazane to ensure your ultra-thin photoresist sticks perfectly to the silicon wafer and doesn't peel off during development.
- **Precision Spin Coater:** Must be highly stable. Because EUV resists need to be only **20 nm to 40 nm thick**, you will be spinning at high speeds (typically 3,000 to 5,000 RPM) and require tight acceleration controls.
- **Programmable Hotplate:** Used for the Post-Apply Bake (PAB) to drive off solvents. It must have precise digital temperature control (within ±0.5°C).

##### 🔬 2. Metrology & Inspection (Pre & Post-Exposure)

You cannot manage what you cannot measure; atomic-scale depth control is critical.

- **Variable-Angle Spectroscopic Ellipsometer:**  
The most critical tool for resist prep. It uses polarized light to measure the exact thickness of your spun photoresist down to the nanometer before you put it into the vacuum chamber.
- **Atomic Force Microscope (AFM) or Optical Profilometer:** Used after development to check the physical step-height and roughness of your printed circuit features.
- **Field-Emission Scanning Electron Microscope (FE-SEM):** Essential for verifying your feature sizes. Standard optical microscopes cannot see patterns printed at 13.5 nm resolutions.

##### 🧪 3. Wet Chemical Development

After you remove the wafer from your custom HHG exposure chamber, you must develop it.

- **Yellow-Lit Chemical Wet Bench:** Standard organic or base-safe chemical hood. EUV and e-beam resists are highly sensitive to blue and UV light; the entire room or hood must use yellow safety sleeves.
- **Glassware & Timer Station:** Simple beakers for your developer (e.g., MIBK:IPA or TMAH) and your stopper/rinse agents (like Pure Isopropyl Alcohol or DI Water).
- **Filtered N2 Blow-Off Gun:** To gently dry the ultra-thin resist features without fracturing them via capillary forces.

##### ⚙️ 4. Pattern Transfer (Optional but needed for functional chips)

Once the resist is developed, you must etch the circuit pattern permanently into your wafer.

- **Reactive Ion Etcher (RIE) / Inductively Coupled Plasma (ICP) Etcher:** A dry plasma etching tool used to transfer the resist pattern into the underlying silicon or silicon dioxide. Wet chemical etching will bleed sideways and ruin your nanoscale resolution.

---

Building a nanometer-scale chip fabrication system without access to an established cleanroom facility means you are setting up a **"mini-fab" benchtop environment** from scratch.

Do not panic. Because you are using a tabletop High-Harmonic Generation (HHG) system with small wafer coupons (e.g., 10mm to 20mm squares), you can handle the chemical processing steps safely using a high-quality **vertical laminar flow hood** with a HEPA filter (creating a localized Class 100 / ISO 5 clean environment) right on a laboratory benchtop. You can buy compact, benchtop spin coaters and digital hotplates online to fulfill your coating and baking needs.

To solve your exposure step, here is the concept blueprint for a **custom, purely mechanical Soft-Contact Mask-to-Wafer Clamping Fixture**. Because you do not have a mechanical designer, this design relies entirely on **simple geometries** (plates and thumbscrews) that any local machine shop can manufacture from a basic sketch.

---

### The Mechanical Principle: The "Kinematic Sandwich"

To prevent crushing your ultra-thin, 50 nm Silicon Nitride mask window, you must never apply uneven or sharp force. The fixture must gently sandwich the mask frame against the wafer coupon using parallel, gravity-assisted, or spring-regulated vertical pressure.

```plaintext
       [Top Clamping Plate (Aluminum)]  <-- Has a central window hole for the EUV beam
                   │
                   ▼
       [3x Leaf Springs / Rubber Shims] <-- Regulates force gently
                   │
                   ▼
  [Silicon Nitride Mask Frame (Face Down)]
  ========================================   <-- SOFT MECHANICAL CONTACT INTERFACE
  [Photoresist-Coated Wafer (Face Up)]
                   ▲
                   │
      [Bottom Base Pedestal (Aluminum)] <-- Securely bolted to your X-Y-Z vacuum stage
```

### Step-by-Step Fixture Design Guide

##### 1. The Bottom Base Pedestal (The Wafer Holder)

- **The Part:** A solid block of **6061-T6 Aluminum** or Stainless Steel that bolts directly onto your vacuum stage.
- **The Feature:** Machine a shallow, recessed square pocket in the center. The pocket should match the exact dimensions of your silicon wafer coupon (e.g., a pocket 15.1 mm x 15.1 mm wide, and 0.5 mm deep for a 15 mm coupon). This ensures the wafer drops in and cannot slide sideways during alignment.

##### 2. The Top Clamping Plate (The Mask Holder)

- **The Part:** A second, thinner aluminum plate that aligns perfectly over the bottom pedestal.
- **The Feature - Central Aperture:** Machine a large clearance hole through the exact center of this top plate. This hole must be wider than your circuit pattern area so that the 13.5 nm EUV beam can pass through unhindered to hit the mask window.
- **The Feature - Alignment Pins:** Drill two or three alignment pin holes on the outer edges of both the top and bottom plates. Press-fit small steel dowel pins into the bottom plate. The top plate slides over these pins, ensuring it can only move perfectly straight up and down, preventing any twisting motion that would scratch your mask or wafer.

##### 3. Protecting the 50 nm Membrane (Force Regulation)

If you tighten a metal screw directly onto a silicon mask frame, the silicon will instantly shatter. You must regulate the clamping force.

- **The Solution:** Adhere thin strips of **Viton rubber** or install miniature **beryllium-copper leaf springs** to the underside of the top clamping plate.
- **How it works:** When you lower the top plate onto the mask, the rubber or leaf springs compress gently. This applies a soft, uniform downward force that holds the mask frame flat against the wafer without rigid point-loading.
- **Vacuum Warning:** Use **Viton** rubber, not standard hardware-store rubber or silicone. Standard rubber outgasses aggressively in a vacuum, which will destroy your HHG system's deep vacuum and ruin the optics.

##### 4. Managing Pre-Exposure Alignment (The Visual Hack)

Since you are aligning manually without a complex internal microscope matrix:

- Mount your long-working-distance digital microscope directly above the fixture on your benchtop *before* putting it into the vacuum chamber.
- Because the Silicon Nitride membrane is transparent to visible light, look through the microscope, look through the central aperture of your top plate, and look right through the mask window.
- Manually nudge the top plate/mask frame relative to the wafer until your alignment markers cross. Once aligned, tighten down the outer perimeter thumbscrews gently, lock the sandwich together, and carefully carry the whole assembled fixture into your vacuum chamber load-lock.

---

### Next Practical Steps

1. **Order Sample Silicon:** Buy a pack of cheap, blank silicon test wafers and practice dicing them into your target coupon sizes (e.g., 15mm x 15mm squares) using a manual diamond scriber.
2. **Draft a 2D Drawing:** Draw a basic top-down and side-profile view of this "sandwich block" on graph paper with your target dimensions. You can take this drawing directly to a local job-shop machinist, and they can cut it out on a manual mill in a single afternoon.
