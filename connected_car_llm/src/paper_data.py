"""
Embedded research paper text for offline ingestion.
Source: "Security Vulnerabilities in Connected Cars: Risks and Mitigation Strategies"
Author: Devanshu Singh, Chitkara University
"""

PAPER_TEXT = """
Security Vulnerabilities in Connected Cars: Risks and Mitigation Strategies
Devanshu Singh — Chitkara University, Punjab, India

Abstract
Connected cars integrate advanced computing systems, Internet connectivity, and wireless communication technologies such as Bluetooth, Wi-Fi, cellular networks, and Vehicle-to-Everything (V2X) communication. While these technologies enhance user convenience, safety, and navigation efficiency, they also introduce significant cybersecurity risks. Modern vehicles rely on Electronic Control Units (ECUs) interconnected via the Controller Area Network (CAN) bus, which was originally designed without security mechanisms. This paper identifies major security vulnerabilities in connected cars, analyzes common attack vectors including CAN bus exploitation, remote keyless entry attacks, and infotainment system breaches, and evaluates existing mitigation strategies. Findings reveal that unsecured CAN protocols, weak authentication, and insufficient wireless encryption represent the primary risk surfaces. Implementation of anomaly-based intrusion detection systems (IDS), network segmentation, secure boot, and end-to-end encryption significantly reduces attack feasibility. The paper concludes that automotive manufacturers must adopt cybersecurity-by-design principles to safeguard future smart mobility ecosystems.

Keywords: Connected Cars, Automotive Cybersecurity, CAN Bus, Intrusion Detection Systems, Vehicle Networks, V2X Communication, ECU Security

1 Introduction

The automotive industry is undergoing a profound digital transformation. Modern vehicles are no longer standalone mechanical systems; they are complex cyber-physical platforms that communicate with external infrastructure, cloud services, and other vehicles. A contemporary connected car may contain over 100 Electronic Control Units (ECUs), millions of lines of software code, and multiple wireless communication interfaces including Bluetooth, Wi-Fi, cellular networks, and Vehicle-to-Everything (V2X) protocols.

While these technological advances deliver substantial benefits — enhanced navigation, predictive maintenance, autonomous driving assistance, and emergency response capabilities — they simultaneously expand the cyberattack surface in ways the automotive industry was not historically designed to address. The Controller Area Network (CAN) bus, the dominant in-vehicle communication protocol since the 1980s, was engineered for reliability and real-time performance, not security. It lacks authentication, encryption, and access control mechanisms, making it inherently vulnerable to exploitation once an attacker gains network access.

High-profile demonstrations of connected car vulnerabilities have illustrated the severity of the problem. In 2015, security researchers Miller and Valasek remotely exploited a Jeep Cherokee through its Uconnect infotainment system, gaining control of steering, brakes, and transmission while the vehicle was in motion on a public highway. This single incident triggered a recall of 1.4 million vehicles and catalyzed regulatory and industry-wide attention to automotive cybersecurity.

This paper presents a systematic analysis of security vulnerabilities in connected cars, organized around the major attack surfaces: in-vehicle networks, wireless communication interfaces, and cloud-connected telematics systems.

2 Related Work

Research into automotive cybersecurity has accelerated significantly since the mid-2010s. Koscher et al. conducted the first comprehensive experimental analysis of automotive attack surfaces, demonstrating that compromising a single in-vehicle ECU could yield control over safety-critical functions including braking and engine management. Their work established the foundational taxonomy of automotive attack vectors that subsequent researchers have built upon.

Checkoway et al. extended this analysis by demonstrating that remote attack vectors — OBD-II diagnostic interfaces, cellular connections, Bluetooth, and even malicious audio CDs — could be chained to achieve full vehicle compromise without physical access. This work fundamentally shifted the threat model for connected vehicles from insider/physical threats to remote adversarial scenarios.

The ISO/SAE 21434 standard, published in 2021, represents the automotive industry's most significant effort to formalize cybersecurity engineering requirements across the vehicle lifecycle, from concept through decommissioning. Complementing this, UNECE WP.29 Regulation 155 mandates cybersecurity management systems for vehicle type approvals in markets including the European Union, Japan, and South Korea.

Recent work has focused on machine learning-based intrusion detection for CAN bus traffic. Longari et al. and Taylor et al. independently demonstrated that anomaly detection models trained on normal CAN message patterns can identify injection attacks with high accuracy, though real-time performance constraints remain a challenge for deployment on resource-constrained ECUs.

3 Connected Car Architecture and Threat Landscape

3.1 In-Vehicle Network Architecture

Modern vehicles employ a hierarchical network architecture. At the lowest layer, the CAN bus interconnects ECUs responsible for powertrain, chassis, body electronics, and safety systems. Higher-bandwidth domains — infotainment, driver assistance, and telematics — increasingly use Ethernet-based protocols such as Automotive Ethernet (100BASE-T1) and FlexRay. Domain controllers and central gateways serve as bridges between these network segments.

The CAN bus protocol operates on a broadcast model: all messages are transmitted to all nodes, and any node can inject messages without authentication. Message integrity is protected only by a cyclic redundancy check (CRC), which detects accidental corruption but provides no protection against deliberate forgery. This architectural limitation is the root cause of the CAN injection vulnerability class.

3.2 External Communication Interfaces

Connected vehicles maintain multiple bidirectional communication channels with external systems:

Telematics Control Units (TCUs) provide LTE/5G cellular connectivity for remote diagnostics, over-the-air (OTA) software updates, emergency services, and connected services platforms.

V2X Communication encompasses Vehicle-to-Vehicle (V2V), Vehicle-to-Infrastructure (V2I), Vehicle-to-Network (V2N), and Vehicle-to-Pedestrian (V2P) protocols, primarily using DSRC (802.11p) or C-V2X standards.

Short-Range Wireless includes Bluetooth (for smartphone integration and key fobs), Wi-Fi (hotspot and diagnostic access), and NFC/UWB (digital keys).

OBD-II Port is a standardized diagnostic interface present in all vehicles manufactured after 1996, providing direct access to vehicle ECUs and the CAN bus.

Each of these interfaces represents a potential entry point for adversaries. The attack surface is compounded by the long operational lifetime of vehicles (typically 10-15 years), which means security vulnerabilities may persist long after patches are available due to OTA update failures or owner non-compliance.

3.3 Threat Actor Classification

Automotive threat actors span a wide spectrum of capability and motivation. Nation-state actors may target vehicle fleets for surveillance or critical infrastructure disruption. Organized criminal groups exploit connected car features for theft — relay attacks against keyless entry systems have become the dominant method of vehicle theft in markets with high adoption of these systems. Researchers and security professionals conduct authorized and unauthorized testing to discover and disclose vulnerabilities. Finally, malicious insiders within the automotive supply chain represent a significant but underappreciated threat vector.

4 Vulnerability Analysis

4.1 CAN Bus Vulnerabilities

The CAN bus is susceptible to several classes of attack that exploit its lack of authentication and broadcast architecture.

Message injection attacks involve an adversary who has gained access to the CAN bus — through the OBD-II port, a compromised ECU, or a gateway — transmitting forged CAN frames. Because receiving ECUs perform no authentication, injected messages are processed identically to legitimate ones. Researchers have demonstrated injection of brake, steering, and engine control commands using this technique.

Denial-of-service attacks exploit the CAN arbitration mechanism: nodes contend for bus access based on message priority (lower arbitration ID = higher priority). An attacker who continuously transmits high-priority messages can starve lower-priority ECUs of bus access, potentially disabling safety-critical systems.

Bus-off attacks leverage the CAN error-handling mechanism. By deliberately inducing transmission errors in a targeted ECU's messages, an attacker can cause the ECU to enter a bus-off state, effectively disabling it. This attack requires only the ability to inject single-bit errors and can permanently disable targeted ECUs.

4.2 Remote Keyless Entry and Immobilizer Attacks

Relay attacks against passive keyless entry and start (PKES) systems have become widespread. These attacks use two relay devices: one amplifies the signal from the key fob inside the victim's home, while another relays it to a second device near the vehicle. The vehicle's receiver detects the amplified signal and unlocks, as it cannot distinguish a relayed signal from a legitimate proximity detection.

Rollback (replay) attacks target older keyless systems that use fixed or insufficiently randomized rolling codes. By jamming the legitimate signal and capturing both key presses, an attacker retains one valid code for later replay. Modern systems using cryptographic rolling codes (e.g., KeeLoq) have partially addressed this, though implementation weaknesses have been exploited in practice.

4.3 Infotainment System Vulnerabilities

Infotainment systems present a large attack surface: they run complex software stacks (often Linux or Android-based), process external data from multiple sources (media, navigation updates, smartphone integration), and are connected to the vehicle's internal network. The 2015 Jeep hack exploited a vulnerability in the D-Bus message broker of the Uconnect system to achieve code execution, then leveraged the connection between the infotainment system and the CAN bus to issue vehicle control commands.

Common vulnerability classes in infotainment systems include memory corruption flaws (buffer overflows, use-after-free), injection vulnerabilities in media parsers, insecure Bluetooth pairing implementations, and weak or default credentials in network-accessible services.

4.4 Telematics and OTA Update Vulnerabilities

Telematics Control Units provide the most direct remote attack surface. Vulnerabilities in TCU firmware — particularly in the modem subsystem or in protocols handling server communications — can provide an entry point reachable over the cellular network. Once the TCU is compromised, its internal network connectivity may allow pivoting to other vehicle systems.

OTA update mechanisms, while essential for timely patching, introduce their own risks. An attacker who can compromise the update server, intercept update communications, or exploit vulnerabilities in the update client can deliver malicious firmware to an entire vehicle fleet. Insufficient code signing, lack of update rollback prevention, and absent boot-time integrity verification compound these risks.

4.5 V2X Communication Vulnerabilities

V2X protocols are designed to enhance safety and traffic efficiency, but their security depends on robust public key infrastructure (PKI). Attacks against V2X include Sybil attacks (one node impersonating multiple vehicles to manipulate traffic flow), replay attacks (retransmitting legitimate safety messages out of context), and GPS spoofing to falsify position data broadcast in V2X messages. The security of DSRC and C-V2X relies on certificate-based authentication, but certificate revocation at scale remains an open engineering challenge.

5 Mitigation Strategies

5.1 CAN Bus Security

Several approaches have been proposed to retrofit security onto the CAN bus. Message Authentication Code (MAC)-based solutions append a cryptographic authenticator to CAN frames, allowing receivers to verify message origin. CANAuth and LiBrA-CAN are representative proposals, though the limited payload size of CAN frames (8 bytes) constrains the size and therefore the security margin of embedded MACs.

Intrusion Detection Systems (IDS) monitor CAN traffic for anomalous patterns indicative of attack. Rule-based IDS check for protocol violations and known attack signatures. Anomaly-based approaches — including statistical models, entropy analysis, and machine learning classifiers — model normal message timing and frequency to detect deviations. Deep learning approaches using LSTM and CNN architectures have shown high detection accuracy on benchmark datasets, though false positive rates and computational overhead remain concerns for real-time deployment.

Network segmentation via domain controllers and firewalled gateways limits lateral movement within the vehicle network. By enforcing strict message filtering between network domains, the impact of a compromised component can be contained.

5.2 Secure Boot and Hardware Security

Secure boot mechanisms verify the cryptographic signature of firmware at each stage of the boot process, preventing execution of unauthorized code. Hardware Security Modules (HSMs) integrated into ECUs provide tamper-resistant storage for cryptographic keys and accelerate cryptographic operations. Modern automotive microcontrollers increasingly integrate HSM functionality on-chip, reducing cost barriers to adoption.

5.3 Wireless Interface Hardening

Mitigation of keyless entry relay attacks includes the use of ultra-wideband (UWB) ranging, which provides centimeter-level distance measurement that relay attacks cannot easily circumvent, and motion sensing in key fobs to disable transmission when the key is stationary. Distance-bounding protocols provide cryptographic guarantees on proximity that are resistant to relay attacks in principle, though practical deployment challenges remain.

For Bluetooth and Wi-Fi interfaces, security measures include enforcing Secure Simple Pairing (SSP) for Bluetooth, disabling unnecessary services, applying firmware patches promptly, and network isolation of infotainment Bluetooth from safety-critical domains.

5.4 Secure OTA Updates

A robust OTA update pipeline requires end-to-end integrity and authenticity protection: updates must be signed by the manufacturer, transmitted over TLS, verified by the vehicle before installation, and subject to version pinning to prevent rollback to vulnerable versions. The Uptane framework provides a reference architecture for secure automotive software updates, incorporating multiple layers of signing and repository mirroring to protect against both server-side compromise and network-level attacks.

5.5 AI-Based Intrusion Detection

Emerging research applies deep learning to vehicle cybersecurity across multiple layers. At the CAN bus layer, recurrent and convolutional neural networks trained on message sequences can detect injection attacks with reported F1 scores exceeding 0.99 on benchmark datasets. At the network layer, federated learning approaches allow anomaly detection models to be trained across a fleet without centralizing sensitive vehicle data. These approaches face deployment challenges including model update logistics, adversarial robustness, and the limited computational resources of automotive-grade hardware.

6 Open Challenges and Future Directions

Despite significant research progress, several fundamental challenges remain unresolved in automotive cybersecurity.

Legacy system security: The majority of vehicles on the road today were designed and manufactured before automotive cybersecurity became an engineering priority. Retrofitting security onto architectures built around unauthenticated CAN bus communication is technically and economically challenging, and the 10-15 year vehicle lifespan means these legacy systems will remain prevalent for decades.

Supply chain security: Modern vehicles incorporate components and software from hundreds of suppliers across global supply chains. Ensuring consistent security standards and coordinating vulnerability disclosure across this ecosystem is a significant governance challenge that technical solutions alone cannot address.

Standardization and regulation: ISO/SAE 21434 and UNECE WP.29 provide frameworks for automotive cybersecurity management, but implementation varies widely. Harmonizing requirements across global markets and ensuring effective enforcement remain ongoing challenges.

Formal verification: Safety-critical automotive software is subject to rigorous functional safety standards (ISO 26262), but equivalent formal methods for security properties are not yet standard practice. Developing tractable approaches to formally verify security properties of automotive software at scale is an important research direction.

Quantum-resistant cryptography: The anticipated availability of cryptographically relevant quantum computers within the operational lifetime of vehicles being manufactured today motivates proactive consideration of post-quantum cryptographic algorithms in automotive PKI and key management systems.

7 Conclusion

Connected car technology represents one of the most consequential deployments of networked computing in terms of direct physical safety implications. This paper has systematically analyzed the principal vulnerability categories in connected vehicles — CAN bus weaknesses, keyless entry attacks, infotainment system exploits, telematics vulnerabilities, and V2X threats — and evaluated the technical measures available to address them.

The fundamental challenge of automotive cybersecurity is architectural: the CAN bus protocol and many vehicle software components were designed for an era of physical isolation, and retrofitting security onto these foundations is inherently limited. Durable security requires cybersecurity-by-design principles to be integrated from the earliest stages of vehicle architecture definition, incorporating hardware security primitives, authenticated communication protocols, network segmentation, and lifecycle security management.

Emerging approaches including AI-based anomaly detection, the Uptane secure update framework, and UWB-based access control represent promising directions, but their effectiveness depends on consistent implementation and rigorous evaluation against adaptive adversaries. Collaboration between automotive manufacturers, Tier 1 suppliers, security researchers, and regulators is essential to develop the standards, tools, and practices necessary to secure the connected vehicle ecosystem as it continues to evolve toward full autonomy.
"""

PAPER_METADATA = {
    "title": "Security Vulnerabilities in Connected Cars: Risks and Mitigation Strategies",
    "author": "Devanshu Singh",
    "institution": "Chitkara University, Punjab, India",
    "keywords": [
        "Connected Cars", "Automotive Cybersecurity", "CAN Bus",
        "Intrusion Detection Systems", "Vehicle Networks", "V2X Communication", "ECU Security"
    ],
    "sections": [
        "Abstract", "Introduction", "Related Work",
        "Connected Car Architecture and Threat Landscape",
        "Vulnerability Analysis", "Mitigation Strategies",
        "Open Challenges and Future Directions", "Conclusion"
    ]
}
