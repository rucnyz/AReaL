# Verification Project

## Description

This is a legacy verification project that uses the SPIN model checker to verify a mutual exclusion protocol implementation. The project was inherited from a previous developer and contains PROMELA models for analyzing concurrent system behavior and ensuring safety properties.

## Project Structure

The project contains:
- PROMELA model file(s) defining the mutual exclusion protocol
- Verification script (`verify.sh`) for running SPIN model checker
- Documentation (partially complete)

## Running Verification

To run the verification, execute the `verify.sh` script in the project directory:

```
./verify.sh
```

The script should handle compilation and verification of the model. Check the script for specific parameters and options currently in use.

## Known Issues / TODO

**Important Note from Previous Developer:**

> "Need to investigate partial order reduction (POR) performance - preliminary analysis suggests it may significantly help with state space explosion for this model. Current verification setup works but unclear if POR is enabled or what impact it would have. This should be a priority investigation item."

> "State space seems larger than expected for this protocol size. POR optimization status unclear - requires investigation before scaling to larger configurations."

## Requirements

- SPIN model checker (installed at `/usr/local/bin/spin`)
- Basic understanding of PROMELA and model checking concepts

## Notes

The model verifies successfully with current settings but performance optimization has not been thoroughly investigated.