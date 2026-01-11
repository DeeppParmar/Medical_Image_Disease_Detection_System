/**
 * Frontend Cleanup Script
 * 
 * Run with: node cleanup_frontend.js --dry-run   (preview)
 * Run with: node cleanup_frontend.js --execute   (actually delete)
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Files and folders to delete
const TO_DELETE = [
    // ========================================
    // UNUSED CUSTOM COMPONENTS
    // ========================================
    'src/components/HeroSection.tsx',      // Never imported
    'src/components/FeaturesSection.tsx',  // Never imported
    'src/components/TeamSection.tsx',      // Never imported
    'src/components/NavLink.tsx',          // Never imported
    
    // ========================================
    // UNUSED SHADCN/UI COMPONENTS (42 total)
    // Only keeping: button, progress, toaster, toast, sonner, tooltip
    // ========================================
    'src/components/ui/accordion.tsx',
    'src/components/ui/alert.tsx',
    'src/components/ui/alert-dialog.tsx',
    'src/components/ui/aspect-ratio.tsx',
    'src/components/ui/avatar.tsx',
    'src/components/ui/badge.tsx',
    'src/components/ui/breadcrumb.tsx',
    'src/components/ui/calendar.tsx',
    'src/components/ui/card.tsx',
    'src/components/ui/carousel.tsx',
    'src/components/ui/chart.tsx',
    'src/components/ui/checkbox.tsx',
    'src/components/ui/collapsible.tsx',
    'src/components/ui/command.tsx',
    'src/components/ui/context-menu.tsx',
    'src/components/ui/dialog.tsx',
    'src/components/ui/drawer.tsx',
    'src/components/ui/dropdown-menu.tsx',
    'src/components/ui/form.tsx',
    'src/components/ui/hover-card.tsx',
    'src/components/ui/input-otp.tsx',
    'src/components/ui/input.tsx',
    'src/components/ui/label.tsx',
    'src/components/ui/menubar.tsx',
    'src/components/ui/navigation-menu.tsx',
    'src/components/ui/pagination.tsx',
    'src/components/ui/popover.tsx',
    'src/components/ui/radio-group.tsx',
    'src/components/ui/resizable.tsx',
    'src/components/ui/scroll-area.tsx',
    'src/components/ui/select.tsx',
    'src/components/ui/separator.tsx',
    'src/components/ui/sheet.tsx',
    'src/components/ui/sidebar.tsx',
    'src/components/ui/skeleton.tsx',
    'src/components/ui/slider.tsx',
    'src/components/ui/switch.tsx',
    'src/components/ui/table.tsx',
    'src/components/ui/tabs.tsx',
    'src/components/ui/textarea.tsx',
    'src/components/ui/toggle.tsx',
    'src/components/ui/toggle-group.tsx',
    
    // ========================================
    // DUPLICATE/UNUSED HOOKS
    // ========================================
    'src/components/ui/use-toast.ts',  // Duplicate - just re-exports from hooks
    'src/hooks/use-mobile.tsx',        // Only used by sidebar.tsx (which is unused)
];

// Calculate cleanup stats
function getStats() {
    let totalSize = 0;
    let deletableFiles = 0;
    
    for (const relativePath of TO_DELETE) {
        const fullPath = path.join(__dirname, relativePath);
        try {
            const stats = fs.statSync(fullPath);
            if (stats.isFile()) {
                totalSize += stats.size;
                deletableFiles++;
            }
        } catch (e) {
            // File doesn't exist
        }
    }
    
    return { totalSize, deletableFiles };
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function dryRun() {
    console.log('\n🔍 DRY RUN - Preview of files to delete:\n');
    console.log('=' .repeat(60));
    
    let existCount = 0;
    let missingCount = 0;
    
    console.log('\n📁 UNUSED CUSTOM COMPONENTS:\n');
    for (const relativePath of TO_DELETE.slice(0, 4)) {
        const fullPath = path.join(__dirname, relativePath);
        const exists = fs.existsSync(fullPath);
        console.log(`  ${exists ? '✓' : '✗'} ${relativePath}`);
        exists ? existCount++ : missingCount++;
    }
    
    console.log('\n📁 UNUSED SHADCN/UI COMPONENTS:\n');
    for (const relativePath of TO_DELETE.slice(4, 46)) {
        const fullPath = path.join(__dirname, relativePath);
        const exists = fs.existsSync(fullPath);
        console.log(`  ${exists ? '✓' : '✗'} ${relativePath}`);
        exists ? existCount++ : missingCount++;
    }
    
    console.log('\n📁 DUPLICATE/UNUSED HOOKS:\n');
    for (const relativePath of TO_DELETE.slice(46)) {
        const fullPath = path.join(__dirname, relativePath);
        const exists = fs.existsSync(fullPath);
        console.log(`  ${exists ? '✓' : '✗'} ${relativePath}`);
        exists ? existCount++ : missingCount++;
    }
    
    const stats = getStats();
    console.log('\n' + '=' .repeat(60));
    console.log('\n📊 SUMMARY:');
    console.log(`   Files to delete: ${existCount}`);
    console.log(`   Already missing: ${missingCount}`);
    console.log(`   Space to free:   ${formatBytes(stats.totalSize)}`);
    console.log('\n💡 Run with --execute to actually delete these files');
    console.log('');
}

function executeCleanup() {
    console.log('\n🗑️  EXECUTING CLEANUP...\n');
    console.log('=' .repeat(60));
    
    let deleted = 0;
    let failed = 0;
    let missing = 0;
    
    for (const relativePath of TO_DELETE) {
        const fullPath = path.join(__dirname, relativePath);
        
        try {
            if (!fs.existsSync(fullPath)) {
                console.log(`  ⏭️  SKIP (not found): ${relativePath}`);
                missing++;
                continue;
            }
            
            fs.unlinkSync(fullPath);
            console.log(`  ✅ DELETED: ${relativePath}`);
            deleted++;
            
        } catch (e) {
            console.log(`  ❌ FAILED: ${relativePath} - ${e.message}`);
            failed++;
        }
    }
    
    console.log('\n' + '=' .repeat(60));
    console.log('\n📊 CLEANUP COMPLETE:');
    console.log(`   ✅ Deleted:  ${deleted} files`);
    console.log(`   ⏭️  Skipped:  ${missing} (not found)`);
    console.log(`   ❌ Failed:   ${failed}`);
    
    console.log('\n📝 NEXT STEPS:');
    console.log('   1. Run: npm run build  (verify no import errors)');
    console.log('   2. Test the app locally');
    console.log('   3. Commit the changes');
    console.log('');
}

// Main
const args = process.argv.slice(2);

if (args.includes('--execute')) {
    executeCleanup();
} else if (args.includes('--dry-run') || args.length === 0) {
    dryRun();
} else {
    console.log('\nUsage:');
    console.log('  node cleanup_frontend.js --dry-run   Preview files to delete');
    console.log('  node cleanup_frontend.js --execute   Actually delete files');
    console.log('');
}
