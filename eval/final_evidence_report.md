# Final Evidence Report — AAA Clinical RAG

Every card below traces: **question → pre-registered gold evidence → retrieved chunk → rank → assessment**.
All retrieved text is the actual indexed chunk text; all gold pages and sections are pre-registered in the frozen gold standards. Nothing here is model-generated.

- Retriever: `abhinand/MedEmbed-base-v0.1` rev `7a90c50263f620dff743eb9794b89a42bfc5d765`, 768-dim, 1330 indexed chunks, top-10
- Original gold SHA-256: `0b8a443b69960bc5ac20311f0010926a2f131bbb5531ccf369f321f59ed2e5c1`
- Held-out gold SHA-256: `67112d0901337591c3e2d1c7f49bc532f800e5835084f6afeb9fa8b5ea8f1c82`
- Production retrieval was **not** modified to produce this report.

---

## Q1 — Screening

*Dataset: **original***

### Question
What are the recommendations for screening for abdominal aortic aneurysm?

### Gold Evidence
- **Guideline:** USPSTF_2019 · **Page:** 1 · **Section:** RECOMMENDATIONS box (B / C / I statements)
  - *Why:* States the 1-time ultrasonography screening recommendation for men 65-75 who have ever smoked, plus the selective-offer and against-routine statements.
- **Guideline:** USPSTF_2019 · **Page:** 2 · **Section:** Summary of Recommendations
  - *Why:* Repeats the graded screening recommendations with the population and net-benefit assessment.
- **Guideline:** ESVS_2024 · **Page:** 20-21 · **Section:** 3.3 Screening for AAA - Recommendation 11
  - *Why:* Recommendation 11: ultrasound screening for early detection of AAA in high risk populations to reduce death from rupture (Class I, Level A).
- **Guideline:** NICE_NG156 · **Page:** 6-7 · **Section:** 1.1.1 - 1.1.5 Identifying and diagnosing AAA
  - *Why:* NHS AAA screening programme recommendations: inform men aged 66 or over, encourage self-referral with risk factors, consider ultrasound for women 70+.
- **Guideline:** SVS_2018 · **Page:** 15 · **Section:** Screening recommendations
  - *Why:* One-time ultrasound screening in men or women 65 to 75 years of age with a history of tobacco use (1/A), plus first-degree relatives and >75 years.

**Required facts** (must satisfy ≥ 2 groups):
- `screening_concept`: screen
- `population_or_modality`: 65 to 75, 65\s*e\s*75, 65-75, aged 66 or over, high risk population, first degree relative …

### Retrieved Evidence

**Rank 1** — **RELEVANT**  ← first relevant
- Guideline: SVS_2018 · Page: 15 · Section: Aneurysm imaging
- chunk_id: `SVS_2018__p15-15__c0016` · Score: 0.8385

> Aneurysm imaging Recommendation Level of recommendation Quality of evidence We recommend using ultrasound, when feasible, as the preferred imaging modality for aneurysm screening and surveillance. 1 A We suggest that the maximum aneurysm diameter derived from computed tomography (CT) imaging should be based on an outer wall to outer wall measurement perpendicular to the path of the aorta. Ungraded Good Practice Statement We recommend a one-time ultrasound screening for AAAs in men or women 65 to 75 years of age with a history of tobacco use. 1 A We suggest ultrasound screening for AAA in first ...

**Rank 2** — not relevant
- Guideline: NICE_NG156 · Page: 29 · Section: Rationale and impact
- chunk_id: `NICE_NG156__p29-29__c0095` · Score: 0.8330

> Men can then self-refer if they feel screening is right for them. Evidence from cross-sectional studies also found that people of Hispanic, African-American and Asian family origin were individually less likely than people of European family origin to have an AAA, so the committee wished to raise awareness of this. Aortic ultrasound is recommended because it is the standard technique used in clinical practice and in the screening programme. It has high diagnostic accuracy, and is associated with lower costs and fewer side effects than CT. People with an AAA diameter of 5.5 cm or larger (inner- ...

**Rank 3** — not relevant
- Guideline: NICE_NG156 · Page: 29 · Section: Rationale and impact
- chunk_id: `NICE_NG156__p29-29__c0094` · Score: 0.8198

> Rationale and impact Identifying asymptomatic abdominal aortic aneurysms Why the committee made the recommendations Recommendations 1.1.1 to 1.1.6 The committee were mindful that the NHS abdominal aortic aneurysm (AAA) screening programme does not cover men under 65 or women of any age. This means some men and all women who are at risk of AAA are not currently screened. The recommendations highlight these groups and specify risk factors significantly associated with AAA that could be used to help with opportunistic screening. There are also men who have no risk factors for AAA and were not see ...

### Evidence Assessment
- Best relevant rank: **1**
- Relevant in Top-1: **YES**
- Relevant in Top-5: **YES**
- Relevant in Top-10: **YES**
- Required facts covered: screening_concept, population_or_modality
- Required facts missing: none
- Facts assessed on: `best_relevant_chunk`
- Gold passages reached: 2/5

---

## Q2 — Repair diameter

*Dataset: **original***

### Question
What AAA diameter is generally associated with consideration of elective repair?

### Gold Evidence
- **Guideline:** ESVS_2024 · **Page:** 26-28 · **Section:** 4.4 Diameter threshold for elective repair - Recommendations 20-23
  - *Why:* Rec 20: men <55 mm not recommended for elective repair. Rec 22: men >=55 mm should be considered. Rec 21/23: women <50 mm not recommended, >=50 mm may be considered.
- **Guideline:** SVS_2018 · **Page:** 19 · **Section:** Timing for intervention
  - *Why:* Elective repair for a fusiform AAA >= 5.5 cm (1/A); repair in women with AAA 5.0-5.4 cm (2/B).
- **Guideline:** NICE_NG156 · **Page:** 16 · **Section:** 1.5.1 When to consider repair
  - *Why:* Consider repair if asymptomatic and 5.5 cm or larger, or >4.0 cm with growth >1 cm in 1 year, or symptomatic.

**Required facts** (must satisfy ≥ 2 groups):
- `diameter_threshold`: 5\.5\s*cm, 55\s*mm, 5\.0\s*cm, 50\s*mm, 5\.4\s*cm, 54\s*mm
- `elective_repair_concept`: repair

### Retrieved Evidence

**Rank 1** — **RELEVANT**  ← first relevant
- Guideline: ESVS_2024 · Page: 26 · Section: ABDOMINAL AORTIC ANEURYSM
- chunk_id: `ESVS_2024__p26-26__c0351` · Score: 0.8431

> On the contrary, based on the NAAASP data it has been suggested to raise the diameter threshold to 60 mm when based on CTA.254 Although it is possible that the threshold should be raised in the future, the GWC does not believe there is suf- ﬁcient support at this time. Nevertheless, the GWC has chosen to issue a new strong negative recommendation of elective repair of AAA < 55 mm, and to downgrade the recommendation on the threshold for considering repair in men (from Class I and LoE A to Class IIa and LoE C) due to the fact that the RCTs underlying this recommendation only showed that it is n ...

**Rank 2** — not relevant
- Guideline: USPSTF_2019 · Page: 4 · Section: Treatment
- chunk_id: `USPSTF_2019__p4-4__c0035` · Score: 0.8325

> The standard of care for elective repair is that patients with an AAA of 5.5 cm or larger in diameter should be referred for surgical intervention with either open repair or EVAR.1 This recommendation is based on RCTs conducted in men. The AAA size needed for surgicalinterventioninwomenmaydiffer.Asaresult,guidelinesfrom the Society for Vascular Surgery recommend repairing AAAs between 5.0 and 5.4 cm in diameter in women.26 However, concerns about poorer surgical outcomes in women, who have more complex anatomy and smaller blood vessels, have led some to caution against lowering the threshold f ...

**Rank 3** — **RELEVANT**
- Guideline: ESVS_2024 · Page: 26 · Section: ABDOMINAL AORTIC ANEURYSM
- chunk_id: `ESVS_2024__p26-26__c0348` · Score: 0.8260

> Multiple papers have reported the mean AAA diameter at the time of rupture, which vary between 75 e 80 mm for men and 67 mm for women.250e252 About 8 e 10% of rAAA operations are done for aneurysms with a diameter < 55 mm. This has been put forward as an argument for lowering the current diameter threshold at which repair is considered. This is, however, a misguided conclusion. Despite small AAA having a very low risk of rupture, their sheer numbers in the population (due to the normal distribution of aortic diameter) make them a sizeable proportion of all operations for rAAA. This is further  ...

### Evidence Assessment
- Best relevant rank: **1**
- Relevant in Top-1: **YES**
- Relevant in Top-5: **YES**
- Relevant in Top-10: **YES**
- Required facts covered: diameter_threshold, elective_repair_concept
- Required facts missing: none
- Facts assessed on: `best_relevant_chunk`
- Gold passages reached: 1/3

---

## Q3 — Surveillance

*Dataset: **original***

### Question
What surveillance strategy is recommended for small abdominal aortic aneurysms?

### Gold Evidence
- **Guideline:** ESVS_2024 · **Page:** 21-22 · **Section:** 4.1 Surveillance of small AAA - Recommendations 13, 14, 15
  - *Why:* RESCAN-based intervals: every 5 years for 25-29 mm, 3 years for 30-39 mm, annually for 40-49 mm, every 6 months for 50-54 mm; separate sex-specific recommendation for women.
- **Guideline:** SVS_2018 · **Page:** 16 · **Section:** Surveillance intervals
  - *Why:* Surveillance imaging at 3-year intervals for 3.0-3.9 cm, 12-month for 4.0-4.9 cm, 6-month for 5.0-5.4 cm; rescreen after 10 years if <3 cm.
- **Guideline:** NICE_NG156 · **Page:** 11-12 · **Section:** 1.2.3 - 1.2.4 Monitoring the risk of rupture
  - *Why:* Offer surveillance with aortic ultrasound to people with an asymptomatic AAA, using the NHS AAA screening programme frequency.

**Required facts** (must satisfy ≥ 2 groups):
- `surveillance_concept`: surveillance, rescreen, monitoring, follow up interval
- `interval_or_frequency`: every five years, every three years, every two years, every six months, every 12 months, 3-year interval …

### Retrieved Evidence

**Rank 1** — not relevant
- Guideline: ESVS_2024 · Page: 6 · Section: Open Surgical Repair
- chunk_id: `ESVS_2024__p6-6__c0098` · Score: 0.8323

> Patients with small abdominal aortic aneurysms, who are either not expected to reach the diameter threshold for repair within their life expectancy, or are unﬁt for repair, or prefer conservative management, should be considered for discontinuation of surveillance. 26. Prior to abdominal aortic aneurysm repair, routine imaging screening of the entire aorta, access and femoropopliteal arteries should be considered. 27. Prior to endovascular abdominal aortic repair, detailed pre-operative procedure planning with computer tomography angiography, including the use of a dedicated post-processing so ...

**Rank 2** — **RELEVANT**  ← first relevant
- Guideline: ESVS_2024 · Page: 22 · Section: ABDOMINAL AORTIC ANEURYSM
- chunk_id: `ESVS_2024__p22-22__c0297` · Score: 0.8289

> Sogaard et al. (2012),179 Rockley et al. (2020)180 Recommendation 14 Changed Women should be considered for imaging surveillance using ultrasound every ﬁve years for a sub-aneurysmal aorta 25 e 29 mm in diameter, every three years for abdominal aortic aneurysms 30 e 39 mm in diameter, annually for aneurysms 40 e 44 mm, and every six months for aneurysms ‡ 45 mm, taking into account life expectancy, suitability for future repair, and patient preferences. Class Level References ToE IIa C Bown et al. (2013)106 Recommendation 15 New Patients with small abdominal aortic aneurysms who are either not ...

**Rank 3** — not relevant
- Guideline: ESVS_2024 · Page: 22 · Section: ABDOMINAL AORTIC ANEURYSM
- chunk_id: `ESVS_2024__p22-22__c0298` · Score: 0.8194

> Patients with small abdominal aortic aneurysms who are either not expected to reach the diameter threshold for repair within their life expectancy, or are unﬁt for repair, or prefer conservative management, should be considered for discontinuation of surveillance. Class Level References IIa C Consensus 4.2. Medical management of patients with small abdominal aortic aneurysms 4.2.1. Cardiovascular risk reduction. Patients with an AAA have a high risk of future cardiovascular events. A systematic review including 21 articles demonstrated a 3% annual risk of cardiovascular death in patients with  ...

### Evidence Assessment
- Best relevant rank: **2**
- Relevant in Top-1: **NO**
- Relevant in Top-5: **YES**
- Relevant in Top-10: **YES**
- Required facts covered: surveillance_concept, interval_or_frequency
- Required facts missing: none
- Facts assessed on: `best_relevant_chunk`
- Gold passages reached: 2/3

---

## Q4 — EVAR indications

*Dataset: **original***

### Question
What are the indications for endovascular aneurysm repair?

### Gold Evidence
- **Guideline:** ESVS_2024 · **Page:** 46 · **Section:** 5.2 Choice of repair modality - Recommendations 65, 66
  - *Why:* Rec 65: for most patients with suitable anatomy and reasonable life expectancy, endovascular repair should be considered the preferred elective modality.
- **Guideline:** ESVS_2024 · **Page:** 54 · **Section:** Recommendation 80 - ruptured AAA
  - *Why:* For patients with a ruptured AAA and suitable anatomy, endovascular repair is recommended as the first line treatment option.
- **Guideline:** NICE_NG156 · **Page:** 17 · **Section:** 1.5.4 - 1.5.5 Standard EVAR
  - *Why:* Consider EVAR for unruptured AAA with abdominal copathology (hostile abdomen, horseshoe kidney, stoma), or where anaesthetic risk/comorbidity contraindicates open surgical repair.
- **Guideline:** NICE_NG156 · **Page:** 19-20 · **Section:** 1.6.1 - 1.6.3 Ruptured aneurysms
  - *Why:* Consider EVAR or open surgical repair for ruptured AAA; open repair if standard EVAR unsuitable.
- **Guideline:** SVS_2018 · **Page:** 29 · **Section:** Patient with a ruptured AAA
  - *Why:* If anatomically feasible, EVAR is recommended over open repair for a ruptured AAA (1/C).

**Required facts** (must satisfy ≥ 2 groups):
- `evar_named`: evar, endovascular
- `indication_criterion`: suitable anatomy, anatomically feasible, anatomical suitability, preferred treatment, first line, contraindicat …

### Retrieved Evidence

**Rank 1** — not relevant
- Guideline: ESVS_2024 · Page: 26 · Section: ABDOMINAL AORTIC ANEURYSM
- chunk_id: `ESVS_2024__p26-26__c0340` · Score: 0.7467

> 4.4. Indications for elective repair The immediate decision about the size at which an aneurysm should be repaired is based on the balance between aneurysm rupture risk (which is fatal in > 80% cases)237 and operative mortality risk of aneurysm repair. Today, with increased life expectancy, it also is necessary to consider the long term prognosis, including durability, surveillance, life expectancy, and the QoL after AAA repair. Furthermore, the patient’s preference is of course key in the decision making (see Chapter 11). The management of fusiform, degenerative aneurysms 40 e 55 mm in diamet ...

**Rank 2** — not relevant
- Guideline: ESVS_2024 · Page: 10 · Section: Table 1-continued
- chunk_id: `ESVS_2024__p10-10__c0180` · Score: 0.7370

> Patients undergoing endovascular repair for a ruptured abdominal aortic aneurysm may be considered for a bifurcated device, in preferencetoanaorto-uni-iliacdevice,wheneveranatomicallysuitable(downgradedtoClassIIb) 85. Patientswithasymptomaticnon-rupturedabdominalaorticaneurysmmaybeconsideredforabriefperiodofrapidassessmentand optimisationfollowedbyurgentrepairunderoptimalconditions(ideallyduringworkinghours)(downgradedtoClassIIb) 116. Patients with complex abdominal aortic aneurysms may be considered for elective repair at a diameter of (cid:1) 55 mm in men and (cid:1)50mminwomen,takingintoacc ...

**Rank 3** — not relevant
- Guideline: ESVS_2024 · Page: 8 · Section: Table 1-continued
- chunk_id: `ESVS_2024__p8-8__c0128` · Score: 0.7263

> For patients undergoing endovascular abdominal aortic aneurysm repair, routine pre-emptive embolisation of the inferior mesenteric artery and lumbar arteries, and non-selective aneurysm sac embolisation is not indicated. 88. For patients treated by endovascular abdominal aortic aneurysm repair who present with asymptomatic non-obstructive mural thrombus formation limited to the main body of the stent graft, intervention or escalation of antithrombotic therapy is not indicated. 124. Hybrid repair, by means of visceral and renal artery re-routing (bypassing) combined with endovascular exclusion  ...

### Evidence Assessment
- Best relevant rank: **none in top-10**
- Relevant in Top-1: **NO**
- Relevant in Top-5: **NO**
- Relevant in Top-10: **NO**
- Required facts covered: evar_named, indication_criterion
- Required facts missing: none
- Facts assessed on: `top10_union`
- Gold passages reached: 0/5

---

## Q5 — Risk factors

*Dataset: **original***

### Question
What are the risk factors associated with abdominal aortic aneurysm?

### Gold Evidence
- **Guideline:** USPSTF_2019 · **Page:** 2 · **Section:** Practice Considerations - Assessment of Risk
  - *Why:* Important risk factors for AAA include older age, male sex, smoking, first-degree relative with AAA; other factors include CAD, cerebrovascular disease, atherosclerosis, hypercholesterolaemia, hypertension.
- **Guideline:** ESVS_2024 · **Page:** 16-17 · **Section:** 2.x Epidemiology and risk factors
  - *Why:* Smoking is the strongest risk factor (OR >3); other risk factors include atherosclerosis, hypertension, ethnicity, and family history.
- **Guideline:** ESVS_2024 · **Page:** 20 · **Section:** 3.3 Screening - high risk groups
  - *Why:* The dominant risk factor for AAA, apart from male sex and age, is smoking; ~75% of AAA cases attributable to smoking.
- **Guideline:** NICE_NG156 · **Page:** 6-7 · **Section:** 1.1.2 - 1.1.3 risk factor lists
  - *Why:* Enumerated risk factors: COPD, coronary/cerebrovascular/peripheral arterial disease, family history of AAA, hyperlipidaemia, hypertension, smoking.

**Required facts** (must satisfy ≥ 2 groups):
- `risk_factor_concept`: risk factor, risk factors, predictor, associated with an increased risk
- `named_factor`: smok, tobacco, family history, first-degree relative, first degree relative, male sex …

### Retrieved Evidence

**Rank 1** — **RELEVANT**  ← first relevant
- Guideline: USPSTF_2019 · Page: 2 · Section: Summary of Recommendations
- chunk_id: `USPSTF_2019__p2-2__c0017` · Score: 0.8335

> Other risk factors include a history of other vascular aneurysms, coronary artery disease, cerebrovascular disease, atherosclerosis, hypercholesterolemia, and hypertension.17-19 Factors associated with a reduced risk include African American race, Hispanic ethnicity, Asian ethnicity, and diabetes.13,20-24 Risk factors for AAA rupture include older age, female sex, smoking, and elevated blood pressure.1 Clinicians should consider the presence of comorbid conditions and not offering screening if patients are unable to undergo surgical intervention or have a reduced life expectancy. Smoking Statu ...

**Rank 2** — **RELEVANT**
- Guideline: NICE_NG156 · Page: 6 · Section: Recommendations
- chunk_id: `NICE_NG156__p6-6__c0040` · Score: 0.7697

> Encourage men aged 66 or over to self-refer to the NHS AAA screening programme if they have not already been screened and they have any of the following risk factors: • chronic obstructive pulmonary disease (COPD) • coronary, cerebrovascular or peripheral arterial disease • family history of AAA • hyperlipidaemia • hypertension • they smoke or used to smoke. Abdominal aortic aneurysm: diagnosis and management (NG156) © NICE 2025. All rights reserved. Subject to Notice of rights (https://www.nice.org.uk/terms-andconditions#notice-of-rights). Page 6 of

**Rank 3** — not relevant
- Guideline: ESVS_2024 · Page: 65 · Section: Recommendation 105
- chunk_id: `ESVS_2024__p65-65__c0839` · Score: 0.7647

> Risk factors associated with persistent or late developing Type 2 endoleaks after endovascular abdominal aorticaneurysmrepair. Riskfactorsconsistentlyreportedinliterature Absenceofcircumferentialthrombusintheaneurysmsacor largeflowlumen455,827e831 Numberofpatentaorticsidebranchesarisingfrom AAA827,831,832 Inferiormesentericarterypatency454,455,828,830,831,833 Numberofpatentlumbararteries>3453e455,829,831e834 Diameterofinferiormesentericartery(cid:1)3mm454,455,834 Diameteroflumbararteries(cid:1)2mm453e455 Anticoagulanttherapy835e839 Riskfactorsinconsistentlyreportedoruncertain

### Evidence Assessment
- Best relevant rank: **1**
- Relevant in Top-1: **YES**
- Relevant in Top-5: **YES**
- Relevant in Top-10: **YES**
- Required facts covered: risk_factor_concept, named_factor
- Required facts missing: none
- Facts assessed on: `best_relevant_chunk`
- Gold passages reached: 3/4

---

## Q6 — Imaging modality

*Dataset: **original***

### Question
What imaging modality is recommended for diagnosis or surveillance of AAA?

### Gold Evidence
- **Guideline:** ESVS_2024 · **Page:** 18-19 · **Section:** 3.2 Imaging - Recommendations 7, 8, 9, 10
  - *Why:* Rec 7: ultrasonography recommended for first line diagnosis and surveillance of small AAA. Rec 9: CTA for treatment planning once threshold met and for diagnosis of rupture.
- **Guideline:** SVS_2018 · **Page:** 15 · **Section:** Imaging modality
  - *Why:* Ultrasound recommended, when feasible, as the preferred imaging modality for aneurysm screening and surveillance (1/A).
- **Guideline:** NICE_NG156 · **Page:** 7-9 · **Section:** 1.1.5, 1.1.10 aortic ultrasound
  - *Why:* Offer an aortic ultrasound where asymptomatic AAA is being considered; report inner-to-inner maximum AP diameter.
- **Guideline:** ESVS_2024 · **Page:** 21-22 · **Section:** 4.1 imaging surveillance using ultrasound
  - *Why:* Recommendations 13/14 specify imaging surveillance using ultrasound for small AAA.

**Required facts** (must satisfy ≥ 2 groups):
- `modality_named`: ultraso, computed tomography, \bcta\b, \bct\b, duplex, magnetic resonance
- `purpose`: diagnos, surveillance, screening, imaging modality, measurement, treatment planning

### Retrieved Evidence

**Rank 1** — not relevant
- Guideline: SVS_2018 · Page: 16 · Section: Aneurysm imaging
- chunk_id: `SVS_2018__p16-16__c0017` · Score: 0.8054

> Aneurysm imaging Recommendation Level of recommendation Quality of evidence If initial ultrasound screening identified an aortic diameter >2.5 cm but <3 cm, we suggest rescreening after 10 years. 2 C We suggest surveillance imaging at 3-year intervals for patients with an AAA between 3.0 and 3.9 cm. 2 C We suggest surveillance imaging at 12-month intervals for patients with an AAA of 4.0 to 4.9 cm in diameter. 2 C We suggest surveillance imaging at 6-month intervals for patients with an AAA between 5.0 and 5.4 cm in diameter. 2 C We recommend a CT scan to evaluate patients thought to have AAA  ...

**Rank 2** — not relevant
- Guideline: USPSTF_2019 · Page: 3 · Section: Treatment
- chunk_id: `USPSTF_2019__p3-3__c0022` · Score: 0.7912

> The primary method of screening for AAA is conventional abdominalduplexultrasonography.26Screeningwithultrasonographyisnoninvasive,issimpletoperform,hashighsensitivity(94%-100%)and specificity (98%-100%) for detecting AAA,1,27-31 and does not exposepatientstoradiation.Computedtomographyisanaccuratetool for identifying AAA; however, it is not recommended as a screening method because of the potential for harms from radiation exposure.1 Physical examination has been used in practice but has low sensitivity (39%-68%) and specificity (75%) and is not recommended for screening.32 Screening Interval ...

**Rank 3** — **RELEVANT**  ← first relevant
- Guideline: ESVS_2024 · Page: 21 · Section: ABDOMINAL AORTIC ANEURYSM
- chunk_id: `ESVS_2024__p21-21__c0282` · Score: 0.7869

> Guirguis-Blake et al. (2014)159 * What can be considered a high risk group varies based on local conditions, such as disease prevalence, life expectancy, and healthcare structure, see Table 6. 3.4. Incidental detection Diagnostic imaging used for the investigation of other pathologies including back or chest pain, abdominal and genitourinary symptoms may also detect an AAA. While US and CT scan are most commonly used, there are other imaging modalities including magnetic resonance imaging (MRI), echocardiography, CT colonography, and spinal imaging that may diagnose an AAA.152,160e164 There is ...

### Evidence Assessment
- Best relevant rank: **3**
- Relevant in Top-1: **NO**
- Relevant in Top-5: **YES**
- Relevant in Top-10: **YES**
- Required facts covered: modality_named, purpose
- Required facts missing: none
- Facts assessed on: `best_relevant_chunk`
- Gold passages reached: 2/4

---

## Q7 — Rupture risk

*Dataset: **original***

### Question
What factors influence the risk of AAA rupture?

### Gold Evidence
- **Guideline:** USPSTF_2019 · **Page:** 2 · **Section:** Assessment of Risk - rupture
  - *Why:* Risk factors for AAA rupture include older age, female sex, smoking, and elevated blood pressure.
- **Guideline:** ESVS_2024 · **Page:** 26-27 · **Section:** 4.4 rupture risk vs diameter, sex, growth
  - *Why:* Rupture risk by diameter (0.3-0.8%/yr below 55 mm in men); women have a fourfold greater rupture risk; rapid growth >10 mm/year associated with greater rupture risk.
- **Guideline:** ESVS_2024 · **Page:** 17 · **Section:** 2.x natural history
  - *Why:* With progressive aneurysm enlargement the risk of AAA rupture increases; quantified by diameter and time.
- **Guideline:** ESVS_2024 · **Page:** 21 · **Section:** 4.1 rupture rates during surveillance
  - *Why:* Annual rupture risk by diameter band in the UK screening programme; women had a fourfold greater rupture risk.
- **Guideline:** ESVS_2024 · **Page:** 28 · **Section:** 4.5 smoking and statins vs rupture risk
  - *Why:* Rupture risk increased two-fold in current smokers; statins may reduce rupture risk.

**Required facts** (must satisfy ≥ 3 groups):
- `rupture_concept`: ruptur
- `risk_concept`: risk, rate
- `influencing_factor`: diameter, size, women, female, smok, growth …

### Retrieved Evidence

**Rank 1** — **RELEVANT**  ← first relevant
- Guideline: USPSTF_2019 · Page: 2 · Section: Summary of Recommendations
- chunk_id: `USPSTF_2019__p2-2__c0017` · Score: 0.8622

> Other risk factors include a history of other vascular aneurysms, coronary artery disease, cerebrovascular disease, atherosclerosis, hypercholesterolemia, and hypertension.17-19 Factors associated with a reduced risk include African American race, Hispanic ethnicity, Asian ethnicity, and diabetes.13,20-24 Risk factors for AAA rupture include older age, female sex, smoking, and elevated blood pressure.1 Clinicians should consider the presence of comorbid conditions and not offering screening if patients are unable to undergo surgical intervention or have a reduced life expectancy. Smoking Statu ...

**Rank 2** — not relevant
- Guideline: ESVS_2024 · Page: 20 · Section: Recommendation 9
- chunk_id: `ESVS_2024__p20-20__c0276` · Score: 0.7952

> The heritability of AAA has been estimated to be 70%,101 and there are reports from several countries of an increased incidence of AAA among ﬁrst degree relatives of patients with AAA.100 In a Swedish population study, a family history of AAA increased the risk of AAA two fold144 and in a large Swedish twin registry study there was a 24% probability that a monozygotic twin of a person with AAA will have the disease.101 Family history of AAA is suggested to be associated with more rapid aneurysm growth and a higher rupture rate145 and rupture may occur at smaller aneurysm diameter and at lower  ...

**Rank 3** — **RELEVANT**
- Guideline: ESVS_2024 · Page: 26 · Section: ABDOMINAL AORTIC ANEURYSM
- chunk_id: `ESVS_2024__p26-26__c0347` · Score: 0.7914

> In a population based screening cohort study, the annual rupture rate for AAAs up to 60 mm was 0.8%.249 Decisive data come from the NAAASP in the UK.116 Screening units use US for surveillance and use the ITI AP diameter.116,117 Rupture rates in men were 0.4% per year for diameters between 50 and 54 mm (which translates to CT diameters between 55 and 59 mm). Studies on rupture risk of small AAAs are displayed in Table 8. Multiple papers have reported the mean AAA diameter at the time of rupture, which vary between 75 e 80 mm for men and 67 mm for women.250e252 About 8 e 10% of rAAA operations  ...

### Evidence Assessment
- Best relevant rank: **1**
- Relevant in Top-1: **YES**
- Relevant in Top-5: **YES**
- Relevant in Top-10: **YES**
- Required facts covered: rupture_concept, risk_concept, influencing_factor
- Required facts missing: none
- Facts assessed on: `best_relevant_chunk`
- Gold passages reached: 3/5

---

## Q8 — Smoking cessation

*Dataset: **original***

### Question
What are the recommendations for smoking cessation in patients with AAA?

### Gold Evidence
- **Guideline:** ESVS_2024 · **Page:** 23-24 · **Section:** 4.2 Recommendations 16, 17
  - *Why:* Rec 16: all patients with AAA should receive cardiovascular risk factor management with smoking cessation. Rec 17: patients with a small AAA are recommended to stop smoking and receive help, to reduce growth rate and rupture risk (Class I, Level B).
- **Guideline:** NICE_NG156 · **Page:** 11 · **Section:** 1.2.1 Reducing the risk of rupture
  - *Why:* Offer a referral to a stop smoking service to people with an AAA who smoke.
- **Guideline:** SVS_2018 · **Page:** 20 · **Section:** Medical management during surveillance
  - *Why:* We recommend smoking cessation to reduce the risk of AAA growth and rupture (1/B).
- **Guideline:** SVS_2018 · **Page:** 13 · **Section:** Preoperative pulmonary evaluation
  - *Why:* We recommend smoking cessation for at least 2 weeks before aneurysm repair.

**Required facts** (must satisfy ≥ 2 groups):
- `smoking_named`: smok, tobacco
- `cessation_action`: cessation, stop smoking, quit, give up smoking, referral to a stop smoking

### Retrieved Evidence

**Rank 1** — **RELEVANT**  ← first relevant
- Guideline: SVS_2018 · Page: 20 · Section: Medical management during the period of
- chunk_id: `SVS_2018__p20-20__c0020` · Score: 0.8074

> Medical management during the period of AAA surveillance Recommendation Level of recommendation Quality of evidence We recommend smoking cessation to reduce the risk of AAA growth and rupture. 1 B We suggest not administering statins, doxycycline, roxithromycin, ACE inhibitors, or angiotensin receptor blockers for the sole purpose of reducing the risk of AAA expansion and rupture. 2 C We suggest not administering beta blocker therapy for the sole purpose of reducing the risk of AAA expansion and rupture. 1 B

**Rank 2** — not relevant
- Guideline: ESVS_2024 · Page: 20 · Section: Recommendation 9
- chunk_id: `ESVS_2024__p20-20__c0273` · Score: 0.8033

> Preventive Services Task Force has recommended AAA screening for men aged 65 e 75 years who have ever smoked, based on the strength of the association between smoking and AAA rather than evidence from RCTs.140 With a recommended screening strategy targeting all men aged 65 years there is currently no need for targeting screening based on smoking status. However, in populations with a decreasing prevalence a more selective high risk screening strategy based on smoking status could be a more effective alternative than general screening. There is limited evidence for screening in women, with the  ...

**Rank 3** — not relevant
- Guideline: USPSTF_2019 · Page: 6 · Section: Treatment
- chunk_id: `USPSTF_2019__p6-6__c0072` · Score: 0.8008

> These organizations do not recommend screening for AAA in men who have never smoked or in women.46 The Society for Vascular Surgery recommends 1-time ultrasonography screening for AAA in all men and women aged 65 to75yearswithahistoryoftobaccouse,men55yearsorolderwith a family history of AAA, and women 65 years or older who have smoked or have a family history of AAA.47 The American College of PreventiveMedicinerecommends1-timescreeninginmenaged65 to 75 years who have ever smoked; it does not recommend routine screening in women.48 Clinical Review & Education US Preventive Services Task Force  ...

### Evidence Assessment
- Best relevant rank: **1**
- Relevant in Top-1: **YES**
- Relevant in Top-5: **YES**
- Relevant in Top-10: **YES**
- Required facts covered: smoking_named, cessation_action
- Required facts missing: none
- Facts assessed on: `best_relevant_chunk`
- Gold passages reached: 2/4

---

## Q9 — Women and screening

*Dataset: **original***

### Question
What are the recommendations for women regarding AAA screening?

### Gold Evidence
- **Guideline:** USPSTF_2019 · **Page:** 1 · **Section:** RECOMMENDATIONS box - women
  - *Why:* C recommendation: against routine screening in women who have never smoked and have no family history; I statement for women 65-75 who have ever smoked or have a family history.
- **Guideline:** USPSTF_2019 · **Page:** 2 · **Section:** Summary of Recommendations - women
  - *Why:* Evidence insufficient for women 65-75 who ever smoked/family history; harms outweigh benefits for women who never smoked with no family history.
- **Guideline:** ESVS_2024 · **Page:** 20 · **Section:** 3.3 Screening in women
  - *Why:* Limited evidence for screening in women, the only RCT underpowered; population screening in women has not been considered given lower prevalence.
- **Guideline:** NICE_NG156 · **Page:** 7 · **Section:** 1.1.3 women aged 70 and over
  - *Why:* Consider an aortic ultrasound for women aged 70 and over with listed risk factors if AAA not already excluded.
- **Guideline:** SVS_2018 · **Page:** 15 · **Section:** Screening recommendations
  - *Why:* One-time ultrasound screening in men or women 65 to 75 years of age with a history of tobacco use.

**Required facts** (must satisfy ≥ 2 groups):
- `women_named`: women, female, woman
- `screening_concept`: screen

### Retrieved Evidence

**Rank 1** — not relevant
- Guideline: USPSTF_2019 · Page: 6 · Section: Treatment
- chunk_id: `USPSTF_2019__p6-6__c0070` · Score: 0.8546

> • Epidemiologic studies on the current prevalence of AAA in the UnitedStates,includinginsubpopulations,wouldhelpinformthe applicability of older population-based screening trials to the current US population. • Well-designedstudies,RCTs,orregistrydataonthethresholdsfor repair of AAA in women may inform the benefits and harms of screeninginwomen,asevidencesuggeststhatAAAsinwomenmay rupture at a smaller size than in men. • Studies examining systems approaches to improving implementation of evidence-based AAA screening in the United States are needed. • Studies examining the efficacy of screening ...

**Rank 2** — not relevant
- Guideline: USPSTF_2019 · Page: 6 · Section: Treatment
- chunk_id: `USPSTF_2019__p6-6__c0071` · Score: 0.8432

> • Studies examining the efficacy of screening and treatment in diverse populations (eg, older adults, women, and racial/ethnic groups) are needed to inform the need for specific recommendations in subpopulations of Americans. Recommendations of Others The American College of Cardiology and the American Heart Association jointly recommend 1-time screening for AAA with physical examination and ultrasonography in men aged 65 to 75 years who have ever smoked or in men 60 years or older who are the sibling or offspring of a person with AAA. These organizations do not recommend screening for AAA in  ...

**Rank 3** — not relevant
- Guideline: USPSTF_2019 · Page: 6 · Section: Treatment
- chunk_id: `USPSTF_2019__p6-6__c0072` · Score: 0.8408

> These organizations do not recommend screening for AAA in men who have never smoked or in women.46 The Society for Vascular Surgery recommends 1-time ultrasonography screening for AAA in all men and women aged 65 to75yearswithahistoryoftobaccouse,men55yearsorolderwith a family history of AAA, and women 65 years or older who have smoked or have a family history of AAA.47 The American College of PreventiveMedicinerecommends1-timescreeninginmenaged65 to 75 years who have ever smoked; it does not recommend routine screening in women.48 Clinical Review & Education US Preventive Services Task Force  ...

**Rank 4** — **RELEVANT**  ← first relevant
- Guideline: ESVS_2024 · Page: 20 · Section: Recommendation 9
- chunk_id: `ESVS_2024__p20-20__c0273` · Score: 0.8340

> Preventive Services Task Force has recommended AAA screening for men aged 65 e 75 years who have ever smoked, based on the strength of the association between smoking and AAA rather than evidence from RCTs.140 With a recommended screening strategy targeting all men aged 65 years there is currently no need for targeting screening based on smoking status. However, in populations with a decreasing prevalence a more selective high risk screening strategy based on smoking status could be a more effective alternative than general screening. There is limited evidence for screening in women, with the  ...

### Evidence Assessment
- Best relevant rank: **4**
- Relevant in Top-1: **NO**
- Relevant in Top-5: **YES**
- Relevant in Top-10: **YES**
- Required facts covered: women_named, screening_concept
- Required facts missing: none
- Facts assessed on: `best_relevant_chunk`
- Gold passages reached: 3/5

---

## Q10 — Open repair vs EVAR

*Dataset: **original***

### Question
What are the differences between open surgical repair and EVAR recommendations?

### Gold Evidence
- **Guideline:** ESVS_2024 · **Page:** 46 · **Section:** 5.2 Recommendations 65 and 66
  - *Why:* Rec 65: EVAR preferred for most patients with suitable anatomy and reasonable life expectancy. Rec 66: open surgical repair preferred for most patients with long life expectancy. The surrounding text contrasts short-term survival benefit of EVAR against long-term durability.
- **Guideline:** NICE_NG156 · **Page:** 17 · **Section:** 1.5.3 - 1.5.5 OSR vs standard EVAR
  - *Why:* Offer open surgical repair unless contraindicated; consider EVAR for abdominal copathology; consider EVAR or conservative management if OSR contraindicated.
- **Guideline:** SVS_2018 · **Page:** 29 · **Section:** Patient with a ruptured AAA
  - *Why:* If anatomically feasible, EVAR recommended over open repair for a ruptured AAA.

**Required facts** (must satisfy ≥ 3 groups):
- `evar_named`: evar, endovascular
- `open_repair_named`: open surgical repair, open repair, open aneurysm repair, \bosr\b, open aortic
- `comparative_or_directive`: prefer, consider, offer, recommend, contraindicat, over open …

### Retrieved Evidence

**Rank 1** — not relevant
- Guideline: NICE_NG156 · Page: 25 · Section: 1 Monitoring frequencies and repair thresholds
- chunk_id: `NICE_NG156__p25-25__c0086` · Score: 0.8127

> What is the effectiveness and cost effectiveness of complex endovascular aneurysm repair (EVAR) versus open surgical repair in people for whom open surgical repair is suitable for: • elective repair of an unruptured AAA or Abdominal aortic aneurysm: diagnosis and management (NG156) © NICE 2025. All rights reserved. Subject to Notice of rights (https://www.nice.org.uk/terms-andconditions#notice-of-rights). Page 25 of

**Rank 2** — not relevant
- Guideline: NICE_NG156 · Page: 40 · Section: Repairing unruptured aneurysms
- chunk_id: `NICE_NG156__p40-40__c0122` · Score: 0.8112

> • has higher net costs and lower net benefits than open surgical repair or • is substantially above the range NICE normally considers to be a cost-effective use of NHS resources. There is a small group of people who have abdominal copathology or other considerations that mean open surgical repair is unsuitable. Examples of copathologies include people who have internal scar-tissue from previous abdominal surgery (a so-called hostile abdomen), people who have a single, fused kidney that is wrapped around the aorta ('horseshoe kidney'), and people who have a stoma. Although there was no evidence ...

**Rank 3** — not relevant
- Guideline: NICE_NG156 · Page: 42 · Section: EVAR and complex EVAR for specific subgroups of people
- chunk_id: `NICE_NG156__p42-42__c0129` · Score: 0.8099

> They show that, while outcomes from EVAR have improved over the last 15 years, outcomes from open surgical repair have also improved by roughly the same amount. This means the difference in outcomes between the two has remained fairly constant. Registries like the National Vascular Registry can provide a useful snapshot of current practice, and the analyses that informed NICE's decision-making made use of data from them. However, they are not designed to evaluate the comparative benefits and harms of different surgical approaches, such as EVAR and open surgical repair. Therefore, they cannot b ...

**Rank 9** — **RELEVANT**  ← first relevant
- Guideline: NICE_NG156 · Page: 17 · Section: None
- chunk_id: `NICE_NG156__p17-17__c0067` · Score: 0.8019

> Consider EVAR or conservative management for people with unruptured AAAs meeting the criteria in recommendation 1.5.1 who have anaesthetic risks and/or medical comorbidities that would contraindicate open surgical repair. Complex endovascular aneurysm repair 1.5.6 If open surgical repair and complex EVAR are both suitable options, only consider complex EVAR if: Abdominal aortic aneurysm: diagnosis and management (NG156) © NICE 2025. All rights reserved. Subject to Notice of rights (https://www.nice.org.uk/terms-andconditions#notice-of-rights). Page 17 of

### Evidence Assessment
- Best relevant rank: **9**
- Relevant in Top-1: **NO**
- Relevant in Top-5: **NO**
- Relevant in Top-10: **YES**
- Required facts covered: evar_named, open_repair_named, comparative_or_directive
- Required facts missing: none
- Facts assessed on: `best_relevant_chunk`
- Gold passages reached: 1/3

---

## H1 — Endoleak management

*Dataset: **heldout***

### Question
How should type 1 and type 3 endoleaks be managed after EVAR?

### Gold Evidence
- **Guideline:** NICE_NG156 · **Page:** 22 · **Section:** 1.8.1 - 1.8.3 Managing endoleaks
  - *Why:* Consider open, endovascular or percutaneous intervention for type 1 and type 3 endoleaks; intervene for type 2 with sac expansion.
- **Guideline:** ESVS_2024 · **Page:** 65-66 · **Section:** 5.3 Type 1 and Type 3 endoleak management
  - *Why:* Type 3 endoleaks expose the sac to aortic pressure so prompt intervention is warranted; proximal extension with fenestrated/branched devices for compromised proximal seal.

**Required facts** (must satisfy ≥ 2 groups):
- `endoleak_named`: endoleak
- `management_action`: intervention, intervene, embolis, percutaneous, open, endovascular …

### Retrieved Evidence

**Rank 1** — **RELEVANT**  ← first relevant
- Guideline: ESVS_2024 · Page: 66 · Section: Recommendation 105
- chunk_id: `ESVS_2024__p66-66__c0852` · Score: 0.7949

> Like Type 1 endoleaks, Type 3 endoleaks expose the aneurysm to direct aortic pressure with subsequent risk of rupture.792 Therefore, prompt intervention is warranted. Management of Type 3a endoleaks is usually 257 Table: Recommendation107 | | | Changed SecondaryinterventionforaType2endoleakafter endovascularabdominalaorticaneurysmrepairshouldonly beconsideredinthepresenceofsignificantaneurysmsac growth(‡10mmcomparedwithbaselineorwiththe smallestdiameterduringfollowupusingthesameimaging modalityandmeasurementmethod),primarilyby endovascularmeans,providedalternativecausesincluding Type1or3endole ...

**Rank 2** — not relevant
- Guideline: ESVS_2024 · Page: 67 · Section: Recommendation 105
- chunk_id: `ESVS_2024__p67-67__c0855` · Score: 0.7911

> straightforward, with the use of an extension limb to bridge the separated components. Occasionally, conversion to AUI may be necessary.718 Type 3b endoleaks may be more challenging, depending on the location of the defect. Often, the exact location is impossible to determine. These endoleaks may be repaired by re-lining the defect, but a tailored approach is necessary. Conversion to open repair with or without graft preservation remains an acceptable option for suitable patients, especially after failed endovascular attempts.423 Exceptionally high frequencies of late Type 3 endoleaks have bee ...

**Rank 3** — not relevant
- Guideline: ESVS_2024 · Page: 78 · Section: Complex AAAs are estimated to constitute about 15 e
- chunk_id: `ESVS_2024__p78-78__c0979` · Score: 0.7848

> Type 1a endoleak and none experienced main body stent migration, aneurysm sac growth, or aneurysm rupture or requiring conversion to OSR through 12 months follow up.999 In a meta-analysis, including 968 patients from eight studies with and without hostile neck, 6% developed a persistent Type 1a endoleak, 0.3% required an additional proximal aortic cuff due to migration of the main graft, and expansion of the aneurysm sac was found in 1.93% after mean six months follow up.1000 The literature on endostaplers is mainly limited to company sponsored reports and long term data on their effectiveness ...

### Evidence Assessment
- Best relevant rank: **1**
- Relevant in Top-1: **YES**
- Relevant in Top-5: **YES**
- Relevant in Top-10: **YES**
- Required facts covered: endoleak_named, management_action
- Required facts missing: none
- Facts assessed on: `best_relevant_chunk`
- Gold passages reached: 2/2

---

## H2 — Post-EVAR imaging follow-up

*Dataset: **heldout***

### Question
What imaging follow-up is recommended after endovascular aneurysm repair?

### Gold Evidence
- **Guideline:** ESVS_2024 · **Page:** 71 · **Section:** Recommendations 113, 115 - post-EVAR imaging
  - *Why:* Early post-operative CTA within 30 days; long term imaging follow up regardless of initial risk stratification.
- **Guideline:** SVS_2018 · **Page:** 47 · **Section:** Postoperative surveillance
  - *Why:* Baseline imaging in the first month after EVAR with contrast-enhanced CT and duplex; repeat at 12 months.
- **Guideline:** NICE_NG156 · **Page:** 21 · **Section:** 1.7.1 - 1.7.3 Surveillance after EVAR
  - *Why:* Enrol people who have had EVAR into a surveillance imaging programme; base frequency on risk of complications.

**Required facts** (must satisfy ≥ 2 groups):
- `evar_named`: evar, endovascular
- `imaging_followup`: imaging, surveillance, computed tomography, \bcta\b, \bct\b, duplex …

### Retrieved Evidence

**Rank 1** — not relevant
- Guideline: ESVS_2024 · Page: 68 · Section: Recommendation 105
- chunk_id: `ESVS_2024__p68-68__c0874` · Score: 0.8041

> Patients who have undergone open surgical repair for abdominal aortic aneurysm may be considered for imaging follow up of the entire aorta and peripheral arteries every ﬁve years. Class Level References ToE IIb C Serizawa et al. (2021),709 Diwan et al. (2000),891 Chaer et al. (2012)892 7.4. Follow up after endovascular aortic repair 7.4.1. Imaging modalities for endovascular aortic repair follow up. The aim of post-operative imaging is to predict or detect complications. Various imaging modalities can be used during EVAR follow up. A list of imaging modalities and their pros and cons is presen ...

**Rank 2** — **RELEVANT**  ← first relevant
- Guideline: ESVS_2024 · Page: 71 · Section: 30 Days post-operative CTA
- chunk_id: `ESVS_2024__p71-71__c0905` · Score: 0.8035

> Patients who have undergone endovascular abdominal aortic aneurysm repair are recommended early post-operative imaging (within 30 days) using computed tomography angiography, to assess the presence of endoleak, component overlap and sealing zone length. Class Level References ToE I B Karthikesalingam et al. 2010),716 Bastos Gonçalves et al. (2013),821 Bastos Gonçalves et al. (2014),822 Baderkhan et al. (2018),823 Geraedts et al. (2021)824 Recommendation 114 Changed Patients who have undergone endovascular abdominal aortic aneurysm repair and have been stratiﬁed as low risk of complications* ba ...

**Rank 3** — **RELEVANT**
- Guideline: ESVS_2024 · Page: 71 · Section: 30 Days post-operative CTA
- chunk_id: `ESVS_2024__p71-71__c0907` · Score: 0.7872

> Antoniou et al. (2020)927 * No endoleak, anatomy within IFU, adequate overlap and seal of  10 mm proximal and distal stent graft apposition to arterial wall. Recommendation 115 New Patients who have undergone endovascular abdominal aortic aneurysm repair are recommended for long term imaging follow up (regardless of initial risk stratiﬁcation), to detect late complications and identify late device failure and disease progression. Class Level References ToE I B Patel et al. (2016),466 Geraedts et al. (2022),917 de Mik et al. (2019),919 Grima et al. (2018),920 Wanken et al. (2020)922 262 Anders ...

### Evidence Assessment
- Best relevant rank: **2**
- Relevant in Top-1: **NO**
- Relevant in Top-5: **YES**
- Relevant in Top-10: **YES**
- Required facts covered: evar_named, imaging_followup
- Required facts missing: none
- Facts assessed on: `best_relevant_chunk`
- Gold passages reached: 2/3

---

## H3 — Antibiotic prophylaxis for repair

*Dataset: **heldout***

### Question
Is antibiotic prophylaxis recommended before abdominal aortic aneurysm repair?

### Gold Evidence
- **Guideline:** ESVS_2024 · **Page:** 33 · **Section:** Recommendation 42
  - *Why:* All patients undergoing open or endovascular AAA repair should receive peri-operative systemic antibiotic prophylaxis.
- **Guideline:** SVS_2018 · **Page:** 33 · **Section:** Perioperative antibiotic prophylaxis
  - *Why:* IV first-generation cephalosporin, or vancomycin if penicillin allergic, within 30 minutes before OSR or EVAR; continue no more than 24 hours.

**Required facts** (must satisfy ≥ 2 groups):
- `antibiotic_named`: antibiotic, cephalosporin, vancomycin
- `prophylaxis_context`: prophyla, before, peri-operative, perioperative, repair

### Retrieved Evidence

**Rank 1** — **RELEVANT**  ← first relevant
- Guideline: ESVS_2024 · Page: 33 · Section: Antiplatelet monotherapy with aspirin or thienopyridines
- chunk_id: `ESVS_2024__p33-33__c0430` · Score: 0.8540

> Patients undergoing elective abdominal aortic aneurysms repair are not recommended to be on dual therapy or oral anticoagulants during the peri-operative period.* Class Level References III C Consensus * See also Recommendation 31. 5.2.2. Antibiotic prophylaxis. Multiple RCTs have shown the beneﬁts of systemic broad spectrum antibiotic prophylaxis during arterial reconstruction.340,341 Contemporary data from the SVS-VQI conﬁrms that prophylactic antibiotics reduce surgical site infections and in hospital mortality following EVAR.287 Therefore, peri-operative intravenous antibiotic prophylaxis  ...

**Rank 2** — not relevant
- Guideline: SVS_2018 · Page: 44 · Section: Late outcomes
- chunk_id: `SVS_2018__p44-45__c0045` · Score: 0.8387

> We recommend antibiotic prophylaxis to prevent graft infection before any dental procedure involving the manipulation of the gingival or periapical region of teeth or perforation of the oral mucosa, including scaling and root canal procedures, for any patient with an aortic prosthesis, whether placed by OSR or EVAR. 1 B We suggest antibiotic prophylaxis before respiratory tract procedures, gastrointestinal or genitourinary procedures, and demotologic or musculoskeletal procedures for any patient with an aortic prothesis if the potential for infection exists or the patient is immunocompromised. ...

**Rank 3** — **RELEVANT**
- Guideline: ESVS_2024 · Page: 33 · Section: Antiplatelet monotherapy with aspirin or thienopyridines
- chunk_id: `ESVS_2024__p33-33__c0432` · Score: 0.8021

> All patients undergoing open or endovascular abdominal aortic aneurysm repair should receive peri-operative systemic antibiotic prophylaxis. Class Level References ToE I A Eldrup-Jorgensen et al. (2020),287 Stewart et al. (2007)340 5.2.3. Anaesthesia and post-operative pain management. Multimodal pain therapy, including the use of a non-opioid regimen should be instituted to maximise the efﬁcacy of pain relief, while minimising the risk of side effects and complications.344 This approach may include the use of epidural analgesia, patient controlled analgesia, or 224 Anders Wanhainen et al. Tab ...

### Evidence Assessment
- Best relevant rank: **1**
- Relevant in Top-1: **YES**
- Relevant in Top-5: **YES**
- Relevant in Top-10: **YES**
- Required facts covered: antibiotic_named, prophylaxis_context
- Required facts missing: none
- Facts assessed on: `best_relevant_chunk`
- Gold passages reached: 2/2

---

## H5 — Permissive hypotension in rupture

*Dataset: **heldout***

### Question
What blood pressure strategy is recommended during transfer of a suspected ruptured aneurysm?

### Gold Evidence
- **Guideline:** ESVS_2024 · **Page:** 49 · **Section:** Recommendation 72
  - *Why:* For patients with a ruptured AAA, a policy of permissive hypotension is recommended (Class I).
- **Guideline:** NICE_NG156 · **Page:** 13 · **Section:** 1.3.6 Emergency transfer
  - *Why:* Consider a restrictive approach to volume resuscitation (permissive hypotension) during emergency transfer.
- **Guideline:** SVS_2018 · **Page:** 29 · **Section:** The patient with a ruptured aneurysm
  - *Why:* Implement hypotensive hemostasis with restriction of fluid resuscitation in the conscious patient.

**Required facts** (must satisfy ≥ 2 groups):
- `hypotension_strategy`: permissive hypotension, hypotensive, restrictive approach to volume, restriction of fluid, fluid resuscitation
- `rupture_context`: ruptur

### Retrieved Evidence

**Rank 1** — **RELEVANT**  ← first relevant
- Guideline: ESVS_2024 · Page: 49 · Section: ABDOMINAL AORTIC ANEURYSM
- chunk_id: `ESVS_2024__p49-49__c0633` · Score: 0.7628

> There is increasing data that BP targets in elderly patients should not be as low as in ﬁt young trauma patients (e.g., soldiers) although most of the data for permissive hypotension was generated from this young group. In the Immediate Management of Patient with Ruptured Aneurysm: Open vs. Endovascular Repair (IMPROVE) trial, the lowest systolic BP was independently associated with the 30 day mortality rate and it was suggested that a minimum systolic BP of 70 mmHg may be too low a threshold for permissive hypotension in patients with rAAA.544 Nevertheless, the recommendation to implement a p ...

**Rank 2** — **RELEVANT**
- Guideline: NICE_NG156 · Page: 13 · Section: Reducing the risk of rupture
- chunk_id: `NICE_NG156__p13-13__c0058` · Score: 0.7479

> Full details of the evidence and the committee's discussion are in evidence review Q: permissive hypotension during transfer of people with ruptured abdominal aortic aneurysm to regional vascular services. 1.4 Predicting and improving surgical outcomes Predicting surgical outcomes for unruptured aneurysms 1.4.1 Consider cardiopulmonary exercise testing when assessing people for elective repair of an asymptomatic abdominal aortic aneurysm (AAA), if it will assist in shared decision making. Abdominal aortic aneurysm: diagnosis and management (NG156) © NICE 2025. All rights reserved. Subject to N ...

**Rank 3** — not relevant
- Guideline: NICE_NG156 · Page: 12 · Section: Reducing the risk of rupture
- chunk_id: `NICE_NG156__p12-12__c0055` · Score: 0.7465

> When making transfer decisions, be aware that people with a confirmed ruptured AAA who have a cardiac arrest and/or have a persistent loss of consciousness have a negligible chance of surviving AAA repair. 1.3.3 For guidance on care of people with a ruptured AAA for whom repair is considered inappropriate, see the NICE guideline on care of dying adults in the last days of life. 1.3.4 When people with a suspected ruptured or symptomatic unruptured AAA have been accepted by a regional vascular service for emergency assessment, ensure that they leave the referring unit within 30 minutes of the de ...

### Evidence Assessment
- Best relevant rank: **1**
- Relevant in Top-1: **YES**
- Relevant in Top-5: **YES**
- Relevant in Top-10: **YES**
- Required facts covered: hypotension_strategy, rupture_context
- Required facts missing: none
- Facts assessed on: `best_relevant_chunk`
- Gold passages reached: 2/3

---

## H6 — Abdominal compartment syndrome

*Dataset: **heldout***

### Question
How should abdominal compartment syndrome be detected after repair of a ruptured aneurysm?

### Gold Evidence
- **Guideline:** ESVS_2024 · **Page:** 56 · **Section:** Recommendation 81
  - *Why:* After open or endovascular treatment for ruptured AAA, post-operative monitoring of intra-abdominal pressure is recommended for early diagnosis of intra-abdominal hypertension or abdominal compartment syndrome.
- **Guideline:** NICE_NG156 · **Page:** 21 · **Section:** 1.6.5 - 1.6.6 Abdominal compartment syndrome
  - *Why:* Be aware people can develop abdominal compartment syndrome after EVAR or open repair of a ruptured AAA; assess if condition does not improve.

**Required facts** (must satisfy ≥ 2 groups):
- `acs_named`: compartment syndrome, intra-abdominal pressure, intra-abdominal hypertension
- `rupture_repair_context`: ruptur, after evar, open surgical repair, post-operative, postoperative

### Retrieved Evidence

**Rank 1** — **RELEVANT**  ← first relevant
- Guideline: NICE_NG156 · Page: 21 · Section: Abdominal compartment syndrome
- chunk_id: `NICE_NG156__p21-21__c0076` · Score: 0.8204

> Abdominal compartment syndrome 1.6.5 Be aware that people can develop abdominal compartment syndrome after EVAR or open surgical repair of a ruptured AAA. 1.6.6 Assess people for abdominal compartment syndrome if their condition does not improve after EVAR or open surgical repair of a ruptured AAA. For a short explanation of why the committee made these recommendations and how they might affect practice, see the rationale and impact section on abdominal compartment syndrome. Full details of the evidence and the committee's discussion are in evidence review U: preventing abdominal compartment s ...

**Rank 2** — not relevant
- Guideline: ESVS_2024 · Page: 54 · Section: None
- chunk_id: `ESVS_2024__p54-54__c0705` · Score: 0.8140

> Abdominal compartment syndrome (ACS) is deﬁned as a sustained IAP > 20 mm Hg (with or without an abdominal perfusion pressure < 60 mmHg) that is associated with new organ dysfunction or failure. Abdominal perfusion pressure is deﬁned as the mean arterial pressure minus the IAP.645,646 IAH and ACS may occur after both open and endovascular repair of rAAA. It is estimated that if measured consistently, an IAP > 20 mmHg occurs in about half the patients after open rAAA repair, and 20% will develop ACS.647 In a metaanalysis of 39 series that were published between 2000 and

**Rank 3** — not relevant
- Guideline: NICE_NG156 · Page: 48 · Section: None
- chunk_id: `NICE_NG156__p48-48__c0146` · Score: 0.7777

> raise awareness of this option. Return to recommendation Abdominal compartment syndrome Why the committee made the recommendations Recommendations 1.6.5 and 1.6.6 There was no evidence relating to preventing or managing abdominal compartment syndrome in people who are having AAA repair. The committee agreed it was important to raise awareness of this potentially life-threatening condition, and made recommendations to highlight that it can occur after both endovascular aneurysm repair and open surgical repair. How the recommendations might affect practice The recommendations will ensure that cl ...

### Evidence Assessment
- Best relevant rank: **1**
- Relevant in Top-1: **YES**
- Relevant in Top-5: **YES**
- Relevant in Top-10: **YES**
- Required facts covered: acs_named, rupture_repair_context
- Required facts missing: none
- Facts assessed on: `best_relevant_chunk`
- Gold passages reached: 1/2

---

## H7 — Preoperative cardiac assessment

*Dataset: **heldout***

### Question
What cardiac assessment is needed before aneurysm repair?

### Gold Evidence
- **Guideline:** SVS_2018 · **Page:** 11-12 · **Section:** Assessment of medical comorbidities - cardiac
  - *Why:* Preoperative resting 12-lead ECG within 30 days; echocardiography for dyspnea of unknown origin; cardiology consultation for significant risk factors; coronary revascularization indications.
- **Guideline:** ESVS_2024 · **Page:** 30 · **Section:** Recommendations 28, 29 - cardiac work up
  - *Why:* Routine referral for pre-operative cardiac work up is not indicated; patients with poor functional capacity or significant clinical risk factors are recommended for cardiac work up.

**Required facts** (must satisfy ≥ 2 groups):
- `cardiac_assessment`: electrocardiogram, \becg\b, echocardiograph, cardiac, coronary, cardiology
- `preoperative_context`: preoperative, pre-operative, before, prior to, work up, planned

### Retrieved Evidence

**Rank 1** — not relevant
- Guideline: ESVS_2024 · Page: 29 · Section: Recommendation 23
- chunk_id: `ESVS_2024__p29-29__c0377` · Score: 0.7391

> intervention with a cardiac risk between 1% and 5%.276 This section provides a broad overview of the relevant factors that should be taken into account in the pre-operative evaluation of patients undergoing aortic repair.277 There is extensive guidance on operative risk assessment and reduction which should be consulted for in depth information.276e283 As a minimum, all patients should undergo a medical history and clinical examination, functional assessment, full blood count, electrolytes and renal function, and electrocardiogram. Additional testing depends upon the individual circumstances o ...

**Rank 2** — **RELEVANT**  ← first relevant
- Guideline: SVS_2018 · Page: 11 · Section: Assessment of medical comorbidities
- chunk_id: `SVS_2018__p11-11__c0009` · Score: 0.7279

> Assessment of medical comorbidities Recommendation Level of recommendation Quality of evidence In patients with active cardiac conditions, including unstable angina, decompensated heart failure, severe vulvular disease, and significant arrythmia, we recommend cardiology consultation before endovascular aneurysm repair (EVAR) or open surgical repair (OSR). 1 B In patients with significant clinical risk factors, such as coronary artery disease, congestive heart failure, cerebrovascular disease, diabetes mellitus, chronic renal insufficiency, and unknown or poor functional capacity (metabolic equ ...

**Rank 3** — not relevant
- Guideline: ESVS_2024 · Page: 32 · Section: Recommendation 23
- chunk_id: `ESVS_2024__p32-32__c0412` · Score: 0.7256

> be advisable and should be considered depending on the degree and quality of nutritional deﬁciency. Recommendation 35 Changed Assessment of pre-operative nutritional status by measuring serum albumin should be considered prior to elective abdominal aortic aneurysm repair, with an albumin level of < 2.8 g/dL as the threshold for pre-operative correction. Class Level References ToE IIa C Inagaki et al. (2017)294 5.1.2.5. Carotid artery assessment. Among more than 15 000 patients operated on for AAA in the US National Quality Improvement Program database the peri-operative stroke risk was 0.8% af ...

### Evidence Assessment
- Best relevant rank: **2**
- Relevant in Top-1: **NO**
- Relevant in Top-5: **YES**
- Relevant in Top-10: **YES**
- Required facts covered: cardiac_assessment, preoperative_context
- Required facts missing: none
- Facts assessed on: `best_relevant_chunk`
- Gold passages reached: 2/2

---

## H8 — Contrast nephropathy and renal protection

*Dataset: **heldout***

### Question
How should renal function be protected in patients receiving contrast for endovascular repair?

### Gold Evidence
- **Guideline:** SVS_2018 · **Page:** 14 · **Section:** Renal considerations
  - *Why:* Preoperative hydration in renal insufficiency; pre/post hydration with saline or dextrose-bicarbonate for contrast-induced nephropathy risk; hold metformin by eGFR threshold.
- **Guideline:** ESVS_2024 · **Page:** 80 · **Section:** Recommendation 126
  - *Why:* Strategy to preserve renal function by dose reduction of iodine contrast media, withdrawal of nephrotoxic drugs and adequate hydration.

**Required facts** (must satisfy ≥ 2 groups):
- `renal_named`: renal, nephropathy, metformin, creatinine, \begfr\b
- `contrast_or_protection`: contrast, hydration, preserve renal function, nephrotoxic

### Retrieved Evidence

**Rank 1** — **RELEVANT**  ← first relevant
- Guideline: ESVS_2024 · Page: 80 · Section: None
- chunk_id: `ESVS_2024__p80-80__c0999` · Score: 0.7994

> to prevent of contrast associated acute kidney injury.1035 In the treatment of complex AAA, preservation of large accessory renal arteries ( 4 mm) is feasible with low complication rates and good patency. It prevents early renal dysfunction and provides higher freedom for midterm renal dysfunction,1036 although so far there is no demonstrated effect on death in early post-operative and follow up period.448 Incorporation of < 4.0 mm renal arteries during f/bEVAR is associated with lower technical success, higher risk of arterial disruption and kidney loss, and lower patency rates at one year,1 ...

**Rank 2** — **RELEVANT**
- Guideline: ESVS_2024 · Page: 80 · Section: None
- chunk_id: `ESVS_2024__p80-80__c1000` · Score: 0.7787

> For patients undergoing endovascular repair of a complex abdominal aortic aneurysm a strategy to preserve renal function by dose reduction of iodine contrast media, withdrawal of nephrotoxic drugs and ensuring adequate hydration should be considered. Class Level References IIa C Consensus Recommendation 127 New For endovascular repair of a complex abdominal aortic aneurysm, preservation of large accessory renal arteries (‡ 4 mm) should be considered. Class Level References ToE IIa C Spanos et al. (2021),448 Torrealba et al. (2022)1036 8.4. Spinal cord ischaemia prevention in complex abdominal  ...

**Rank 3** — not relevant
- Guideline: ESVS_2024 · Page: 31 · Section: Recommendation 23
- chunk_id: `ESVS_2024__p31-31__c0407` · Score: 0.7669

> 2 or 3; eGFR < 60 but > 30 mL/min/1.73 m2) should be adequately hydrated before AAA repair, especially when intra-arterial contrast media will be used.310 Currently, no clear effective strategies besides appropriate hydration to prevent post-operative acute kidney injury after AAA repair have been demonstrated.312 Hence, urine output should always be monitored peri-operatively. Recommendation 33 Unchanged Assessment of pre-operative kidney function by measuring serum creatinine and estimating glomerular ﬁltration rate is recommended prior to elective abdominal aortic aneurysm repair, with refe ...

### Evidence Assessment
- Best relevant rank: **1**
- Relevant in Top-1: **YES**
- Relevant in Top-5: **YES**
- Relevant in Top-10: **YES**
- Required facts covered: renal_named, contrast_or_protection
- Required facts missing: none
- Facts assessed on: `best_relevant_chunk`
- Gold passages reached: 2/2

---

## H11 — Internal iliac artery preservation

*Dataset: **heldout***

### Question
Should internal iliac artery flow be preserved during aneurysm repair?

### Gold Evidence
- **Guideline:** SVS_2018 · **Page:** 23 · **Section:** EVAR - iliac considerations
  - *Why:* Preservation of flow to at least one internal iliac artery; branch endograft devices in suitable anatomy; stage bilateral internal iliac occlusion by 1-2 weeks.
- **Guideline:** ESVS_2024 · **Page:** 85 · **Section:** Recommendation 138
  - *Why:* For common iliac artery aneurysm repair requiring internal iliac embolisation or ligation, occlusion of the proximal main stem is preferred.

**Required facts** (must satisfy ≥ 2 groups):
- `internal_iliac_named`: internal iliac, hypogastric
- `flow_action`: preservation, preserve, perfusion, embolis, occlusion, ligation …

### Retrieved Evidence

**Rank 1** — not relevant
- Guideline: ESVS_2024 · Page: 84 · Section: None
- chunk_id: `ESVS_2024__p84-84__c1054` · Score: 0.7889

> Recommendation 137 Unchanged Preserving blood ﬂow to at least one internal iliac artery during open surgical and endovascular repair of iliac artery aneurysms is recommended. Class Level References ToE I C Bosanquet et al. (2017),782 Jean-Baptiste et al. (2014)1100 275 Table: Recommendation136 | | | Changed Thechoiceofsurgicaltechniqueforiliacarteryaneurysm repairshouldbeconsideredbasedonindividualpatientand lesioncharacteristics. Class Level References ToE | | | IIa | B | Bucketal.(2015),1086 Yangetal.(2020),1089 Illuminatietal.(2009),1090 Giaquintaetal.(2018),1091 Kouvelosetal.(2016)1093 | R ...

**Rank 2** — not relevant
- Guideline: ESVS_2024 · Page: 84 · Section: None
- chunk_id: `ESVS_2024__p84-84__c1050` · Score: 0.7716

> 13.9%.826,1076 Post-procedural sexual dysfunction, bowel ischaemia and SCI are rarely reported. The likelihood and severity of these complications are more frequent with bilateral IIA occlusion,782,1093,1101 but cannot easily be predicted. Therefore, preservation of blood ﬂow to at least one and ideally both IIAs is recommended if it does not compromise the primary treatment goal of aneurysm exclusion. The availability of IBDs now allows preservation of IIA ﬂow in most cases with suitable anatomy, leading to a reduced incidence of buttock claudication in the treatment of aorto-iliac AAAs and I ...

**Rank 3** — not relevant
- Guideline: ESVS_2024 · Page: 37 · Section: Recommendation 48
- chunk_id: `ESVS_2024__p37-37__c0477` · Score: 0.7590

> At least one internal iliac artery (IIA) should be preserved or re-implanted when possible, to provide sufﬁ- cient perfusion of pelvic organs and to reduce the risk of buttock claudication and colonic ischaemia.396e399 Suture ligation of the inferior mesenteric artery (IMA) should be performed at its origin from the aneurysm sac to preserve left colic collaterals. There is no evidence in the literature to support routine re-implantation of a patent IMA, but it may be considered in selected cases of suspected insufﬁcient visceral perfusion with risk of colonic ischaemia, for example if the supe ...

**Rank 10** — **RELEVANT**  ← first relevant
- Guideline: SVS_2018 · Page: 23 · Section: EVAR
- chunk_id: `SVS_2018__p23-23__c0023` · Score: 0.7320

> EVAR Recommendation Level of recommendation Quality of evidence We recommend preservation of flow to at least one internal iliac artery. 1 A We recommend using Food and Drug Administration (FDA)- approved branch endograft devices in anatomically suitable patients to maintain perfusion to at least one internal iliac artery. 1 A We recommend staging bilateral internal iliac artery occlusion by at least 1 to 2 weeks if required for EVAR. 1 A We suggest renal artery or superior mesenteric artery (SMA) angioplasty and stenting for selected patients with symptomatic disease before EVAR or OSR. 2 C W ...

### Evidence Assessment
- Best relevant rank: **10**
- Relevant in Top-1: **NO**
- Relevant in Top-5: **NO**
- Relevant in Top-10: **YES**
- Required facts covered: internal_iliac_named, flow_action
- Required facts missing: none
- Facts assessed on: `best_relevant_chunk`
- Gold passages reached: 1/2

---

## H13 — Venous thromboembolism prophylaxis

*Dataset: **heldout***

### Question
What thromboprophylaxis is recommended for patients undergoing aneurysm repair?

### Gold Evidence
- **Guideline:** ESVS_2024 · **Page:** 36 · **Section:** Recommendation 48
  - *Why:* All patients undergoing elective AAA repair at risk of post-operative venous thromboembolism should be considered for thromboprophylaxis.
- **Guideline:** SVS_2018 · **Page:** 39 · **Section:** Thromboprophylaxis
  - *Why:* Intermittent pneumatic compression and early ambulation for all patients undergoing OSR or EVAR; heparin for moderate-to-high VTE risk.

**Required facts** (must satisfy ≥ 2 groups):
- `vte_named`: thromboprophylaxis, venous thromboembolism, \bvte\b, pneumatic compression, deep vein
- `repair_context`: repair, \bosr\b, evar, undergoing, post-operative, postoperative

### Retrieved Evidence

**Rank 1** — **RELEVANT**  ← first relevant
- Guideline: SVS_2018 · Page: 39 · Section: Prophylaxis for deep venous thrombosis
- chunk_id: `SVS_2018__p39-39__c0039` · Score: 0.8412

> Prophylaxis for deep venous thrombosis Recommendation Level of recommendation Quality of evidence We recommend thromboprophylaxis that includes intermittent pneumatic compression and early ambulation for all patients undergoing OSR or EVAR. 1 A We suggest thromboprophylaxis with unfractionated or lowmolecular-weight heparin for patients undergoing aneurysm repair at moderate to high risk for venous thromboembolism and low risk for bleeding. 2 C

**Rank 2** — not relevant
- Guideline: ESVS_2024 · Page: 6 · Section: Open Surgical Repair
- chunk_id: `ESVS_2024__p6-6__c0099` · Score: 0.7717

> All patients undergoing elective abdominal aortic aneurysm repair and deemed at risk of post-operative venous thromboembolism should be considered for thromboprophylaxis. 50. For open abdominal aortic aneurysm repair, the choice of midline vs. transverse or transperitoneal vs. retroperitoneal abdominal incision should be considered based on surgeon preference and patient factors. 56. For endovascular abdominal aortic aneurysm repair, device selection should be considered based on aorto-iliac anatomy and the availability of unbiased long term durability data. Continued Continued 197 Table: Tabl ...

**Rank 3** — not relevant
- Guideline: ESVS_2024 · Page: 35 · Section: Recommendation 45
- chunk_id: `ESVS_2024__p35-35__c0455` · Score: 0.7708

> The authors suggested a selective VTE prophylaxis strategy, based on the risk of development of postoperative VTE, in patients undergoing major vascular surgery.377 Similarly, a recent systematic review and metaanalysis of RCTs assessing the role of thromboprophylaxis after vascular surgery, including eight studies with a total of 3 130 patients, demonstrated a non-signiﬁcant trend towards a lower risk of post-operative deep venous thrombosis (DVT) (RR 0.3, p ¼ .060) and pulmonary embolism (RR 0.17, p ¼ .17) among patients receiving VTE prophylaxis.

### Evidence Assessment
- Best relevant rank: **1**
- Relevant in Top-1: **YES**
- Relevant in Top-5: **YES**
- Relevant in Top-10: **YES**
- Required facts covered: vte_named, repair_context
- Required facts missing: none
- Facts assessed on: `best_relevant_chunk`
- Gold passages reached: 2/2

---

## H14 — Anaesthesia and analgesia

*Dataset: **heldout***

### Question
What analgesia is recommended for people having open surgical repair?

### Gold Evidence
- **Guideline:** NICE_NG156 · **Page:** 19 · **Section:** 1.5.8 Anaesthesia and analgesia
  - *Why:* Consider epidural analgesia in addition to general anaesthesia for people having open surgical repair of an unruptured AAA.
- **Guideline:** ESVS_2024 · **Page:** 34 · **Section:** Recommendation 43
  - *Why:* Patients undergoing elective open AAA repair may be considered for peri-operative epidural analgesia or catheter based continuous wound analgesia.
- **Guideline:** SVS_2018 · **Page:** 41 · **Section:** Postoperative pain control
  - *Why:* Multimodality treatment including epidural analgesia for postoperative pain control after OSR.

**Required facts** (must satisfy ≥ 2 groups):
- `analgesia_named`: epidural, analgesia, anaesthesia, anesthesia, pain control
- `open_repair_context`: open, \bosr\b

### Retrieved Evidence

**Rank 1** — not relevant
- Guideline: SVS_2018 · Page: 32 · Section: Choice of anesthetic technique and agent
- chunk_id: `SVS_2018__p32-32__c0033` · Score: 0.7190

> Choice of anesthetic technique and agent Recommendation Level of recommendation Quality of evidence We recommend general endotracheal anesthesia for patients undergoing open aneurysm repair. 1 A Table: Choice of anesthetic technique and agent Recommendation Level of Quality of recommendation evidence We recommend general endotracheal anesthesia for patients 1 A undergoing open aneurysm repair. Recommendation | Level of recommendation | Quality of evidence We recommend general endotracheal anesthesia for patients undergoing open aneurysm repair. | 1 | A

**Rank 2** — not relevant
- Guideline: NICE_NG156 · Page: 47 · Section: EVAR and complex EVAR for specific subgroups of people
- chunk_id: `NICE_NG156__p47-47__c0144` · Score: 0.7134

> Tranexamic acid is used in varying degrees across the NHS, but it is not standard practice for people with ruptured or symptomatic AAAs who are being transferred before surgery. Return to recommendations Anaesthesia and analgesia during ruptured aneurysm repair Why the committee made the recommendation Recommendation 1.6.4 No evidence was identified on the optimal use of anaesthesia and analgesia in people having open surgical repair or EVAR of a ruptured AAA. The committee agreed, based on their knowledge and experience, that general anaesthesia alone is widely accepted as best practice for o ...

**Rank 3** — **RELEVANT**  ← first relevant
- Guideline: ESVS_2024 · Page: 34 · Section: Antiplatelet monotherapy with aspirin or thienopyridines
- chunk_id: `ESVS_2024__p34-34__c0442` · Score: 0.6957

> While the data regarding the preferred method of anaesthesia in elective EVAR are limited, the GWC ﬁnd it to be appropriate, in the light of current evidence and the proven beneﬁt of local anaesthesia in ruptured EVAR, to issue a weak recommendation favouring locoregional anaesthesia over general anaesthesia in elective settings. Recommendation 43 Changed Patients undergoing elective open abdominal aortic aneurysm repair may be considered for peri-operative epidural analgesia or catheter based continuous wound analgesia, to maximise pain relief and minimise early postoperative complications. C ...

### Evidence Assessment
- Best relevant rank: **3**
- Relevant in Top-1: **NO**
- Relevant in Top-5: **YES**
- Relevant in Top-10: **YES**
- Required facts covered: analgesia_named, open_repair_context
- Required facts missing: none
- Facts assessed on: `best_relevant_chunk`
- Gold passages reached: 3/3

---
