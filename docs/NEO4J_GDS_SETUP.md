# Neo4j Graph Data Science (GDS) Setup Guide

## Overview
The RAGsistant system can work with or without the Neo4j Graph Data Science library. If GDS is not available, the system automatically falls back to a standard Cypher-based community detection method.

## Current Status
- ✅ **Automatic Detection**: System automatically detects if GDS is available
- ✅ **Fallback Method**: Uses connected components analysis when GDS is not available
- ✅ **No Breaking Changes**: System works seamlessly with or without GDS

## Installing Neo4j GDS (Optional)

### For Neo4j Desktop
1. Open Neo4j Desktop
2. Select your database
3. Go to "Plugins" tab
4. Find "Graph Data Science Library"
5. Click "Install"
6. Restart the database

### For Neo4j Server
1. Download the GDS plugin from: https://neo4j.com/docs/graph-data-science/current/installation/
2. Place the JAR file in the `plugins/` directory of your Neo4j installation
3. Add to `neo4j.conf`:
   ```
   dbms.security.procedures.unrestricted=gds.*
   dbms.security.procedures.allowlist=gds.*
   ```
4. Restart Neo4j server

### For Docker
Add the GDS plugin to your Docker setup:
```dockerfile
FROM neo4j:5.0
COPY neo4j-graph-data-science-*.jar /plugins/
ENV NEO4J_dbms_security_procedures_unrestricted=gds.*
ENV NEO4J_dbms_security_procedures_allowlist=gds.*
```

## Verification
To check if GDS is installed and working:
```cypher
CALL gds.version() YIELD gdsVersion
```

## Benefits of GDS vs Fallback

### With GDS (Advanced)
- **Leiden Algorithm**: More sophisticated community detection
- **Louvain Algorithm**: Alternative community detection method
- **Better Performance**: Optimized graph algorithms
- **More Features**: Additional graph analytics capabilities

### Without GDS (Fallback)
- **Connected Components**: Basic but effective community detection
- **No Dependencies**: Works with any Neo4j installation
- **Reliable**: Uses standard Cypher queries
- **Automatic**: No configuration required

## Current Implementation
The system automatically:
1. Checks if GDS is available using `CALL gds.version()`
2. Uses GDS algorithms (Leiden/Louvain) if available
3. Falls back to connected components analysis if GDS is not available
4. Provides the same API regardless of which method is used

## Troubleshooting

### Common Issues
1. **"Procedure not found" errors**: GDS is not installed or not properly configured
2. **Permission errors**: Check `neo4j.conf` security settings
3. **Version compatibility**: Ensure GDS version matches Neo4j version

### Solutions
- The system automatically handles GDS unavailability
- No manual intervention required
- Community detection will work in both cases
- Check logs for which method is being used

## Performance Comparison

### Small Graphs (< 1000 entities)
- **GDS**: Minimal performance difference
- **Fallback**: Adequate performance, simpler implementation

### Large Graphs (> 1000 entities)
- **GDS**: Significantly better performance and quality
- **Fallback**: May be slower but still functional

## Recommendation
- **For Development**: Fallback method is sufficient
- **For Production**: Consider installing GDS for better performance
- **For Large Datasets**: GDS is recommended but not required
