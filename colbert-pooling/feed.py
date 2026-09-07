#!/usr/bin/env python3
"""Feed 100 sample documents to Vespa.

All embedding (non-pooled, pooled, pooled-binary) is done inside Vespa by
the PoolingColBertEmbedder.  This script only sends text.

Usage:
    uv run python feed.py [--endpoint http://localhost:8080]
    # Or generate JSONL for vespa-cli:
    uv run python feed.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import time

import requests

# ---------------------------------------------------------------------------
# 100 sample documents – diverse topics for meaningful retrieval testing
# ---------------------------------------------------------------------------

DOCUMENTS: list[dict[str, str]] = [
    {"title": "Solar System Overview", "text": "The solar system consists of the Sun and the celestial bodies that orbit it, including eight planets, their moons, dwarf planets, asteroids, and comets. The four inner planets are rocky worlds, while the outer planets are gas and ice giants."},
    {"title": "Photosynthesis", "text": "Photosynthesis is the biological process by which green plants and certain organisms convert light energy into chemical energy stored in glucose. It takes place primarily in chloroplasts using chlorophyll pigments."},
    {"title": "Machine Learning Basics", "text": "Machine learning is a branch of artificial intelligence where computers learn patterns from data without being explicitly programmed. Common approaches include supervised learning, unsupervised learning, and reinforcement learning."},
    {"title": "The Roman Empire", "text": "The Roman Empire was one of the largest empires in ancient history, spanning from Britain to Mesopotamia at its peak. It profoundly influenced Western civilisation through its legal system, architecture, engineering, and language."},
    {"title": "DNA Structure", "text": "Deoxyribonucleic acid is a double helix molecule that carries the genetic instructions for the development and functioning of all known living organisms. Watson and Crick described its structure in 1953."},
    {"title": "Ocean Currents", "text": "Ocean currents are continuous movements of seawater driven by wind, temperature differences, salinity, and the Earth's rotation. The Gulf Stream carries warm water from the tropics toward northern Europe."},
    {"title": "Neural Networks", "text": "Artificial neural networks are computational models inspired by the human brain. They consist of layers of interconnected nodes that process information using weighted connections adjusted during training."},
    {"title": "The French Revolution", "text": "The French Revolution began in 1789 with the storming of the Bastille and resulted in profound social and political changes. It ended the absolute monarchy and established principles of citizenship and rights."},
    {"title": "Quantum Mechanics", "text": "Quantum mechanics describes the behaviour of matter and energy at the atomic and subatomic level. Key principles include wave-particle duality, the uncertainty principle, and quantum entanglement."},
    {"title": "Climate Change", "text": "Climate change refers to long-term shifts in global temperatures and weather patterns, largely driven by human activities since the Industrial Revolution, primarily the burning of fossil fuels."},
    {"title": "Protein Folding", "text": "Protein folding is the physical process by which a polypeptide chain acquires its functional three-dimensional structure. Misfolded proteins are associated with diseases like Alzheimer's and Parkinson's."},
    {"title": "The Silk Road", "text": "The Silk Road was an ancient network of trade routes connecting China to the Mediterranean, facilitating the exchange of goods, ideas, technologies, and cultural practices across civilisations for centuries."},
    {"title": "Blockchain Technology", "text": "Blockchain is a distributed ledger technology that records transactions across many computers so that records cannot be altered retroactively. It underpins cryptocurrencies like Bitcoin."},
    {"title": "Plate Tectonics", "text": "Plate tectonics is the theory that Earth's outer shell is divided into large plates that float on a semi-fluid layer of the mantle. Movements at plate boundaries cause earthquakes, volcanic activity, and mountain building."},
    {"title": "The Human Immune System", "text": "The immune system is a complex network of cells, tissues, and organs that work together to defend the body against pathogens. It includes innate immunity and adaptive immunity with memory cells."},
    {"title": "Renewable Energy Sources", "text": "Renewable energy comes from naturally replenished sources such as solar, wind, hydroelectric, geothermal, and biomass. These sources produce far fewer greenhouse gas emissions than fossil fuels."},
    {"title": "The Industrial Revolution", "text": "The Industrial Revolution began in Britain in the late 18th century and transformed manufacturing through mechanisation, steam power, and later electricity. It caused massive urbanisation and social change."},
    {"title": "General Relativity", "text": "Einstein's theory of general relativity describes gravity as the curvature of spacetime caused by mass and energy. It predicts phenomena such as gravitational lensing, time dilation, and black holes."},
    {"title": "Antibiotic Resistance", "text": "Antibiotic resistance occurs when bacteria evolve mechanisms to survive exposure to antibiotics. Overuse and misuse of antibiotics in medicine and agriculture accelerate this global health threat."},
    {"title": "The Renaissance", "text": "The Renaissance was a cultural movement from the 14th to 17th century originating in Italy. It marked a renewed interest in classical art, literature, science, and humanist philosophy."},
    {"title": "Deep Learning Architectures", "text": "Deep learning uses multi-layered neural networks to learn hierarchical representations of data. Key architectures include convolutional networks for images, recurrent networks for sequences, and transformers for language."},
    {"title": "The Water Cycle", "text": "The water cycle describes the continuous movement of water through evaporation, condensation, precipitation, and collection. It is driven by solar energy and gravity, redistributing water across the planet."},
    {"title": "CRISPR Gene Editing", "text": "CRISPR-Cas9 is a revolutionary gene editing technology that allows scientists to precisely modify DNA sequences. It has applications in treating genetic diseases, improving crops, and basic research."},
    {"title": "Ancient Egyptian Civilisation", "text": "Ancient Egypt flourished along the Nile River for over three thousand years. The Egyptians built pyramids, developed hieroglyphic writing, and made advances in medicine, engineering, and astronomy."},
    {"title": "Natural Language Processing", "text": "Natural language processing enables computers to understand, interpret, and generate human language. Modern NLP relies on transformer models pre-trained on large text corpora."},
    {"title": "Volcanic Eruptions", "text": "Volcanic eruptions occur when magma from beneath the Earth's crust reaches the surface. They can cause lava flows, ash clouds, pyroclastic flows, and global temperature changes from aerosol emissions."},
    {"title": "The Cold War", "text": "The Cold War was a geopolitical tension between the United States and the Soviet Union from 1947 to 1991. It involved proxy wars, nuclear arms races, and ideological competition without direct military conflict."},
    {"title": "Superconductivity", "text": "Superconductivity is a phenomenon where certain materials exhibit zero electrical resistance below a critical temperature. Superconductors are used in MRI machines, particle accelerators, and maglev trains."},
    {"title": "Coral Reef Ecosystems", "text": "Coral reefs are diverse underwater ecosystems built by colonies of coral polyps. They support roughly 25 percent of all marine species despite covering less than one percent of the ocean floor."},
    {"title": "Autonomous Vehicles", "text": "Self-driving cars use sensors, cameras, lidar, and artificial intelligence to navigate without human input. They promise safer roads but face challenges in edge cases, regulation, and public trust."},
    {"title": "The Periodic Table", "text": "The periodic table organises chemical elements by atomic number and electron configuration. Mendeleev's original 1869 table predicted the existence and properties of elements not yet discovered."},
    {"title": "Space Exploration", "text": "Space exploration began with the launch of Sputnik in 1957 and has included crewed Moon landings, robotic Mars rovers, and the International Space Station. Private companies now contribute significantly."},
    {"title": "Microbiome Research", "text": "The human microbiome comprises trillions of microorganisms living in and on the body. Research links gut microbiome composition to digestion, immunity, mental health, and chronic diseases."},
    {"title": "The Byzantine Empire", "text": "The Byzantine Empire was the continuation of the Eastern Roman Empire, lasting from the fall of Rome in 476 AD until 1453. Constantinople was its capital and a centre of trade, art, and scholarship."},
    {"title": "Computer Vision", "text": "Computer vision is the field of AI that enables machines to interpret visual information from images and videos. Applications include facial recognition, medical imaging analysis, and autonomous navigation."},
    {"title": "Earthquake Seismology", "text": "Seismology studies earthquakes and the propagation of seismic waves through the Earth. Seismographs measure ground motion and help scientists understand Earth's internal structure and predict hazards."},
    {"title": "The Enlightenment", "text": "The Enlightenment was an intellectual movement of the 17th and 18th centuries emphasising reason, science, and individual rights. Thinkers like Locke, Voltaire, and Kant shaped modern democratic thought."},
    {"title": "Fusion Energy", "text": "Nuclear fusion is the process that powers the Sun, combining light atomic nuclei to release vast amounts of energy. Achieving controlled fusion on Earth would provide a nearly limitless clean energy source."},
    {"title": "Biodiversity Loss", "text": "Biodiversity loss refers to the decline of species variety on Earth, driven by habitat destruction, pollution, overexploitation, invasive species, and climate change. It threatens ecosystem stability."},
    {"title": "Recommender Systems", "text": "Recommender systems suggest relevant items to users based on preferences and behaviour. Collaborative filtering and content-based filtering are the two primary approaches."},
    {"title": "The Circulatory System", "text": "The circulatory system transports blood, oxygen, and nutrients throughout the body via the heart, arteries, veins, and capillaries. It also removes metabolic waste products."},
    {"title": "Cryptography Fundamentals", "text": "Cryptography secures communication through mathematical techniques. Modern methods include symmetric encryption like AES, asymmetric encryption like RSA, and hash functions like SHA-256."},
    {"title": "The Mongol Empire", "text": "The Mongol Empire, founded by Genghis Khan in 1206, became the largest contiguous land empire in history. It facilitated trade, communication, and cultural exchange across Eurasia."},
    {"title": "Gravitational Waves", "text": "Gravitational waves are ripples in spacetime caused by accelerating massive objects. First detected by LIGO in 2015, they opened a new window for observing the universe."},
    {"title": "Deforestation", "text": "Deforestation is the large-scale removal of forest cover, primarily for agriculture, logging, and urban expansion. It contributes to carbon emissions, soil erosion, and loss of habitat."},
    {"title": "Information Retrieval", "text": "Information retrieval is the science of finding relevant documents from a large collection given a user query. It underpins search engines and uses techniques like inverted indexes, TF-IDF, and neural ranking."},
    {"title": "The Nervous System", "text": "The nervous system coordinates the body's actions by transmitting electrical signals between the brain, spinal cord, and peripheral nerves. Neurons communicate via synapses using neurotransmitters."},
    {"title": "Cloud Computing", "text": "Cloud computing delivers computing services over the internet, including servers, storage, databases, networking, and software. Major providers include AWS, Azure, and Google Cloud Platform."},
    {"title": "The Ottoman Empire", "text": "The Ottoman Empire was a vast state founded in 1299 that at its height controlled Southeast Europe, Western Asia, and North Africa. It lasted until 1922, influencing art, law, and governance."},
    {"title": "Dark Matter and Dark Energy", "text": "Dark matter and dark energy make up about 95 percent of the universe's total mass-energy content. Dark matter provides gravitational scaffolding for galaxies, while dark energy drives cosmic expansion."},
    {"title": "Soil Science", "text": "Soil is a complex mixture of minerals, organic matter, water, and air that supports plant growth. Soil health affects agricultural productivity, water filtration, and carbon storage."},
    {"title": "Graph Neural Networks", "text": "Graph neural networks extend deep learning to graph-structured data, learning representations of nodes, edges, and entire graphs. Applications include molecular property prediction and social network analysis."},
    {"title": "The American Civil War", "text": "The American Civil War from 1861 to 1865 was fought between the Union and the Confederacy over slavery and states' rights. It resulted in the abolition of slavery and the preservation of the Union."},
    {"title": "Semiconductor Physics", "text": "Semiconductors are materials with electrical conductivity between conductors and insulators. Silicon-based semiconductors are the foundation of modern electronics, from transistors to integrated circuits."},
    {"title": "Pollinator Decline", "text": "Pollinators like bees, butterflies, and bats are declining due to pesticides, habitat loss, and disease. Their loss threatens food production since many crops depend on animal pollination."},
    {"title": "Transformer Models", "text": "The transformer architecture uses self-attention mechanisms to process sequences in parallel, replacing recurrent models. It powers modern language models like BERT, GPT, and ColBERT for retrieval."},
    {"title": "The Respiratory System", "text": "The respiratory system facilitates gas exchange, bringing oxygen into the body and removing carbon dioxide. Air passes through the trachea to the bronchi and into the alveoli in the lungs."},
    {"title": "Distributed Systems", "text": "Distributed systems are networked computers that coordinate to achieve a common goal. Challenges include consistency, availability, partition tolerance, and the CAP theorem constrains their design."},
    {"title": "The Ming Dynasty", "text": "The Ming Dynasty ruled China from 1368 to 1644, known for its strong centralised government, the construction of the Forbidden City, and maritime expeditions led by Admiral Zheng He."},
    {"title": "Exoplanet Discovery", "text": "Exoplanets are planets orbiting stars outside our solar system. Thousands have been discovered using transit photometry and radial velocity methods, with some in habitable zones."},
    {"title": "Wetland Ecosystems", "text": "Wetlands are transitional areas between land and water that provide critical ecosystem services. They filter pollutants, buffer floods, store carbon, and support diverse wildlife populations."},
    {"title": "Attention Mechanisms", "text": "Attention mechanisms allow neural networks to focus on relevant parts of the input when producing output. Self-attention computes relevance scores between all positions in a sequence."},
    {"title": "The Digestive System", "text": "The digestive system breaks down food into nutrients that the body uses for energy, growth, and repair. It includes the mouth, oesophagus, stomach, small intestine, and large intestine."},
    {"title": "Cybersecurity Threats", "text": "Common cybersecurity threats include malware, phishing, ransomware, denial-of-service attacks, and supply chain compromises. Defence requires layered security, patching, and user awareness."},
    {"title": "The Inca Empire", "text": "The Inca Empire was the largest pre-Columbian empire in the Americas, stretching along the Andes mountains. They built Machu Picchu, developed an extensive road system, and used quipu for record keeping."},
    {"title": "Stellar Evolution", "text": "Stars form from collapsing clouds of gas and dust, undergo nuclear fusion on the main sequence, and end their lives as white dwarfs, neutron stars, or black holes depending on their mass."},
    {"title": "Freshwater Scarcity", "text": "Freshwater scarcity affects billions of people worldwide due to population growth, pollution, and climate change. Only about three percent of Earth's water is fresh, and most is locked in ice."},
    {"title": "Vector Databases", "text": "Vector databases are specialised systems for storing and querying high-dimensional vector embeddings. They use approximate nearest neighbour algorithms like HNSW for fast similarity search."},
    {"title": "The Endocrine System", "text": "The endocrine system regulates body functions through hormones secreted by glands such as the pituitary, thyroid, adrenal glands, and pancreas. It controls metabolism, growth, and reproduction."},
    {"title": "Containerisation Technology", "text": "Containers package applications with their dependencies for consistent deployment across environments. Docker popularised containerisation, while Kubernetes orchestrates container workloads at scale."},
    {"title": "The Viking Age", "text": "The Viking Age spanned from roughly 793 to 1066 AD, during which Norse seafarers from Scandinavia explored, traded, and settled across Europe, the North Atlantic, and even North America."},
    {"title": "Neutrino Physics", "text": "Neutrinos are nearly massless subatomic particles that interact very weakly with matter. Billions pass through your body every second. Their study reveals information about nuclear reactions in stars."},
    {"title": "Urban Heat Islands", "text": "Urban heat islands are metropolitan areas significantly warmer than surrounding rural areas due to human activities, dark surfaces absorbing heat, and reduced vegetation and evapotranspiration."},
    {"title": "Embedding Models", "text": "Embedding models map discrete objects like words or documents to continuous vector spaces where semantic similarity is preserved as geometric distance. ColBERT produces multi-vector embeddings per token."},
    {"title": "The Skeletal System", "text": "The skeletal system provides structural support, protects organs, enables movement, stores minerals, and produces blood cells in bone marrow. Adults have 206 bones connected by joints and ligaments."},
    {"title": "API Design Principles", "text": "Good API design follows principles of consistency, discoverability, and minimal surprise. RESTful APIs use HTTP verbs and resource-based URLs, while GraphQL offers flexible query capabilities."},
    {"title": "The Mughal Empire", "text": "The Mughal Empire ruled much of the Indian subcontinent from 1526 to 1857. It was known for cultural achievements including the Taj Mahal, miniature painting, and a sophisticated administrative system."},
    {"title": "Cosmic Microwave Background", "text": "The cosmic microwave background is the residual radiation from the Big Bang, filling the universe as a nearly uniform glow at about 2.7 Kelvin. Its tiny fluctuations map the early universe's density variations."},
    {"title": "Glacier Retreat", "text": "Glaciers worldwide are retreating due to rising temperatures. Their loss contributes to sea level rise, alters freshwater supplies for millions of people, and affects ecosystems that depend on glacial meltwater."},
    {"title": "Late Interaction Retrieval", "text": "Late interaction models like ColBERT represent queries and documents as sets of token embeddings and score relevance via MaxSim. This balances the effectiveness of cross-encoders with the efficiency of bi-encoders."},
    {"title": "The Musculoskeletal System", "text": "Muscles work with bones and joints to produce movement. Skeletal muscles are voluntary and contract in response to nerve signals. Tendons attach muscles to bones, while ligaments connect bones to each other."},
    {"title": "DevOps Practices", "text": "DevOps combines software development and IT operations to shorten the development lifecycle. Key practices include continuous integration, continuous delivery, infrastructure as code, and monitoring."},
    {"title": "The Aztec Empire", "text": "The Aztec Empire flourished in central Mexico from the 14th to 16th century. Tenochtitlan, their capital built on an island in Lake Texcoco, was one of the largest cities in the world at the time."},
    {"title": "Black Hole Physics", "text": "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape. They form from the collapse of massive stars and are detected through gravitational effects."},
    {"title": "Air Quality and Health", "text": "Poor air quality from particulate matter, ozone, and nitrogen oxides causes respiratory and cardiovascular diseases. Indoor and outdoor air pollution contributes to millions of premature deaths annually."},
    {"title": "Retrieval Augmented Generation", "text": "Retrieval augmented generation combines information retrieval with language model generation. The system first retrieves relevant documents, then conditions the language model's output on the retrieved context."},
    {"title": "The Lymphatic System", "text": "The lymphatic system is a network of vessels, nodes, and organs that maintains fluid balance, absorbs fats from digestion, and plays a key role in immune defence by filtering pathogens."},
    {"title": "Database Indexing", "text": "Database indexes are data structures that improve the speed of data retrieval operations. B-tree indexes handle range queries efficiently, while hash indexes excel at exact-match lookups."},
    {"title": "The Han Dynasty", "text": "The Han Dynasty ruled China from 206 BC to 220 AD and is considered a golden age. It established the Silk Road trade, advanced paper making, and created a civil service examination system."},
    {"title": "Magnetar Stars", "text": "Magnetars are neutron stars with extremely powerful magnetic fields, a trillion times stronger than Earth's. They produce intense bursts of X-rays and gamma rays and are the strongest magnets known in the universe."},
    {"title": "Plastic Pollution", "text": "Plastic pollution accumulates in oceans, rivers, and soil, harming wildlife through ingestion and entanglement. Microplastics enter food chains and have been found in human blood and organs."},
    {"title": "Semantic Search", "text": "Semantic search goes beyond keyword matching to understand the meaning and intent behind queries. It uses dense vector representations and neural models to find conceptually relevant results."},
    {"title": "The Renal System", "text": "The kidneys filter blood to remove waste products and excess fluid, producing urine. They also regulate electrolyte balance, blood pressure through the renin-angiotensin system, and red blood cell production."},
    {"title": "Microservices Architecture", "text": "Microservices decompose applications into small, independently deployable services that communicate via APIs. This approach improves scalability and allows teams to develop and deploy services independently."},
    {"title": "The Gupta Empire", "text": "The Gupta Empire in India from 320 to 550 AD is called the Golden Age of India. It saw advances in mathematics including the concept of zero, astronomy, literature, and art."},
    {"title": "Gamma-Ray Bursts", "text": "Gamma-ray bursts are the most energetic electromagnetic events in the universe, lasting from milliseconds to several hours. They are associated with supernovae and neutron star mergers."},
    {"title": "Ocean Acidification", "text": "Ocean acidification occurs as seawater absorbs excess carbon dioxide from the atmosphere, lowering its pH. This threatens shell-forming organisms, coral reefs, and entire marine food webs."},
    {"title": "Dense Retrieval", "text": "Dense retrieval uses learned vector representations to match queries and documents. Unlike sparse methods such as BM25, dense models capture semantic relationships beyond exact term overlap."},
    {"title": "The Reproductive System", "text": "The reproductive system enables organisms to produce offspring. In humans, it involves the ovaries and uterus in females and the testes in males, with hormones coordinating reproductive cycles."},
    {"title": "Observability in Software", "text": "Observability encompasses logging, metrics, and tracing to understand system behaviour in production. It helps engineers diagnose issues, track performance, and improve reliability of distributed systems."},
]

assert len(DOCUMENTS) == 100, f"Expected 100 documents, got {len(DOCUMENTS)}"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Feed 100 documents to Vespa")
    parser.add_argument("--endpoint", default="http://localhost:8080")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write JSONL instead of feeding to Vespa")
    args = parser.parse_args()

    feed_records = []
    for i, doc in enumerate(DOCUMENTS):
        rec = {
            "put": f"id:doc:doc::{i}",
            "fields": {
                "doc_id": str(i),
                "title": doc["title"],
                "text": doc["text"],
            },
        }
        feed_records.append(rec)

    if args.dry_run:
        from pathlib import Path
        out = Path("ext") / "feed.jsonl"
        out.parent.mkdir(exist_ok=True)
        with open(out, "w") as f:
            for rec in feed_records:
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote {len(feed_records)} documents to {out}")
        print("Feed with:  vespa feed ext/feed.jsonl")
    else:
        t0 = time.time()
        for i, rec in enumerate(feed_records):
            url = f"{args.endpoint}/document/v1/doc/doc/docid/{i}"
            resp = requests.post(url, json={"fields": rec["fields"]}, timeout=60)
            if resp.status_code not in (200, 201):
                print(f"  WARN doc {i}: {resp.status_code} {resp.text[:200]}")
            if (i + 1) % 25 == 0:
                print(f"  [{i+1:3d}/100] ({time.time()-t0:.1f}s)")
        print(f"Fed {len(feed_records)} documents in {time.time()-t0:.1f}s")
        print("Vespa computes all ColBERT embeddings (non-pooled, pooled, binary) at indexing time.")


if __name__ == "__main__":
    main()
