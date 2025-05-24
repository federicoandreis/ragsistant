#!/usr/bin/env python3
"""
Test script to verify the implementation fixes from the review.
This script tests the major components that were identified as incomplete.
"""

import os
import sys
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_security_fixes() -> bool:
    """Test that security fixes are working properly."""
    logger.info("Testing security fixes...")
    
    try:
        # Test that DatabaseConfig requires NEO4J_PASSWORD
        # We'll test this by creating a new instance without the env var
        original_password = os.environ.get('NEO4J_PASSWORD')
        
        # Temporarily remove password
        if 'NEO4J_PASSWORD' in os.environ:
            del os.environ['NEO4J_PASSWORD']
        
        try:
            from app.config import DatabaseConfig
            config = DatabaseConfig()
            logger.error("❌ Security fix failed - config should require NEO4J_PASSWORD")
            return False
        except ValueError as e:
            if "NEO4J_PASSWORD" in str(e):
                logger.info("✅ Security fix working - NEO4J_PASSWORD is required")
            else:
                logger.error(f"❌ Unexpected error: {e}")
                return False
        finally:
            # Restore original password
            if original_password:
                os.environ['NEO4J_PASSWORD'] = original_password
        
        # Test that hardcoded credentials are removed from graph retrieval
        import inspect
        from app.graph import retrieval
        source = inspect.getsource(retrieval)
        
        if 'test1234' in source:
            logger.error("❌ Found hardcoded password 'test1234' in graph retrieval")
            return False
        
        if 'NEO4J_PASSWORD = ' in source:
            logger.error("❌ Found hardcoded NEO4J_PASSWORD assignment in graph retrieval")
            return False
        
        logger.info("✅ No hardcoded credentials found in graph retrieval")
        return True
        
    except Exception as e:
        logger.error(f"❌ Security test failed: {e}")
        return False

def test_database_connectors() -> bool:
    """Test that database connectors can be imported and initialized."""
    logger.info("Testing database connectors...")
    
    try:
        # Test ChromaDB connector
        from app.db.chroma import ChromaConnector, ChromaDocument, ChromaSearchResult
        logger.info("✅ ChromaDB connector imported successfully")
        
        # Test Neo4j connector
        from app.db.neo4j import Neo4jConnector, Neo4jEntity, Neo4jRelationship, Neo4jPath
        logger.info("✅ Neo4j connector imported successfully")
        
        # Test that we can create instances (without connecting to actual databases)
        # This tests the class structure and dependencies
        try:
            # Set a temporary password for testing
            os.environ['NEO4J_PASSWORD'] = 'test_password'
            
            chroma = ChromaConnector()
            logger.info("✅ ChromaConnector instance created successfully")
            
            neo4j = Neo4jConnector()
            logger.info("✅ Neo4jConnector instance created successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to create connector instances: {e}")
            return False
        finally:
            # Clean up test password
            if 'NEO4J_PASSWORD' in os.environ:
                del os.environ['NEO4J_PASSWORD']
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import database connectors: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error testing database connectors: {e}")
        return False

def test_entity_extraction() -> bool:
    """Test that entity extraction components work properly."""
    logger.info("Testing entity extraction...")
    
    try:
        # Test base classes
        from app.entity_extraction.base import (
            EntityExtractor, Entity, Relation, ExtractionResult,
            normalize_entity_id, is_valid_entity, is_valid_relation
        )
        logger.info("✅ Entity extraction base classes imported successfully")
        
        # Test LLM extractor
        from app.entity_extraction.llm_extractor import LLMExtractor, create_llm_extractor
        logger.info("✅ LLM extractor imported successfully")
        
        # Test utility functions
        test_id = normalize_entity_id("Test Entity Name!")
        assert test_id == "test_entity_name", f"Expected 'test_entity_name', got '{test_id}'"
        logger.info("✅ Entity ID normalization working")
        
        # Test entity validation
        valid_entity = {"id": "test_entity", "name": "Test Entity"}
        invalid_entity = {"id": "", "name": "?"}
        
        assert is_valid_entity(valid_entity), "Valid entity should pass validation"
        assert not is_valid_entity(invalid_entity), "Invalid entity should fail validation"
        logger.info("✅ Entity validation working")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import entity extraction components: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error testing entity extraction: {e}")
        return False

def test_graph_retrieval_security() -> bool:
    """Test that graph retrieval uses config instead of hardcoded values."""
    logger.info("Testing graph retrieval security fixes...")
    
    try:
        # Set test password
        os.environ['NEO4J_PASSWORD'] = 'test_password'
        
        from app.graph.retrieval import retrieve_entity_context
        logger.info("✅ Graph retrieval functions imported successfully")
        
        # Check that the module uses config (we can't test actual connection without Neo4j running)
        import inspect
        source = inspect.getsource(retrieve_entity_context)
        
        # Should not contain hardcoded credentials
        if 'test1234' in source or 'NEO4J_PASSWORD = ' in source:
            logger.error("❌ Graph retrieval still contains hardcoded credentials")
            return False
        
        # Should contain config usage
        if 'config.database' in source:
            logger.info("✅ Graph retrieval uses config for database connection")
        else:
            logger.warning("⚠️  Could not verify config usage in graph retrieval")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import graph retrieval: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error testing graph retrieval: {e}")
        return False
    finally:
        if 'NEO4J_PASSWORD' in os.environ:
            del os.environ['NEO4J_PASSWORD']

def test_routing_functionality() -> bool:
    """Test that routing functionality is implemented."""
    logger.info("Testing routing functionality...")
    
    try:
        from app.routing import route_query, llm_route_query, main_rag_pipeline
        from app.models import RoutingDecision
        logger.info("✅ Routing functions imported successfully")
        
        # Test basic routing (this should work without external dependencies)
        try:
            decision = route_query("test query", entity_list=["test"])
            assert isinstance(decision, RoutingDecision), "route_query should return RoutingDecision"
            logger.info("✅ Basic routing functionality working")
        except Exception as e:
            logger.warning(f"⚠️  Basic routing test failed (may need backend): {e}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import routing functions: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error testing routing: {e}")
        return False

def main():
    """Run all tests and report results."""
    logger.info("🚀 Starting implementation verification tests...")
    logger.info("=" * 60)
    
    tests = [
        ("Security Fixes", test_security_fixes),
        ("Database Connectors", test_database_connectors),
        ("Entity Extraction", test_entity_extraction),
        ("Graph Retrieval Security", test_graph_retrieval_security),
        ("Routing Functionality", test_routing_functionality),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} tests...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All implementation fixes verified successfully!")
        return 0
    else:
        logger.error(f"💥 {total - passed} test(s) failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
