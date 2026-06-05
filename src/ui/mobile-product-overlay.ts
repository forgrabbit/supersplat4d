import { Button, Container } from '@playcanvas/pcui';

import { Events } from '../events';
import hiddenSvg from './svg/hidden.svg';
import shownSvg from './svg/shown.svg';
import hostImage from '../../images/host.png';
import ruanImage from '../../images/ruan.png';
import skiImage from '../../images/ski.png';

type ProductOverlayInfo = {
    image: string;
    price: string;
    originalPrice?: string;
    label?: string;
    benefitTitle?: string;
    benefitText?: string;
    statusText?: string;
    extraText?: string;
};

const createSvg = (svgString: string) => {
    const decodedStr = decodeURIComponent(svgString.substring('data:image/svg+xml,'.length));
    return new DOMParser().parseFromString(decodedStr, 'image/svg+xml').documentElement as unknown as SVGElement;
};

const productOverlayMap: Record<string, ProductOverlayInfo> = {
    'host_demo.sog4d': {
        image: hostImage,
        price: '4405',
        originalPrice: '5887',
        label: '\u5230\u624B\u4EF7',
        benefitTitle: '520\u793C\u9047\u5B63',
        benefitText: '\u9001\u597D\u793C \u4E0A\u4EAC\u4E1C',
        statusText: '\u5DF2\u4EAB\uff1A\u5B98\u65B9\u76F4\u964D 25%',
        extraText: '\u53EF\u518D\u4EAB\uff1A\u6EE1200\u51CF10'
    },
    'ruan_demo.sog4d': {
        image: ruanImage,
        price: '1537',
        originalPrice: '1547',
        label: '\u5230\u624B\u4EF7',
        benefitTitle: '\u56FD\u5BB6\u8865\u8D34',
        benefitText: '\u767D\u67616\u671F\u514D\u606F',
        statusText: '\u5DF2\u4EAB\uff1A\u6EE1200\u5DF2\u51CF10',
        extraText: '\u53EF\u518D\u4EAB\uff1A80\u5143\u4F18\u60E0\u5238'
    },
    'ski_demo.sog4d': {
        image: skiImage,
        price: '4147.2',
        originalPrice: '5120',
        label: '\u5230\u624B\u4EF7',
        benefitTitle: '\u56FD\u5BB6\u8865\u8D34',
        benefitText: '\u767D\u67616\u671F\u514D\u606F',
        statusText: '\u5DF2\u4EAB\uff1A\u6EE11\u5957\u4EAB 8.6 \u6298',
        extraText: '\u53EF\u518D\u4EAB\uff1A80\u5143\u4F18\u60E0\u5238'
    }
};

const getModelKey = (name: string | null | undefined) => {
    if (!name) {
        return '';
    }

    return name.split('/').pop()?.split('\\').pop()?.toLowerCase() ?? '';
};

class MobileProductOverlay extends Container {
    private readonly thumbWrap: HTMLDivElement;
    private readonly thumbCard: HTMLDivElement;
    private readonly thumbImage: HTMLImageElement;
    private readonly priceBar: HTMLDivElement;
    private readonly labelElement: HTMLSpanElement;
    private readonly priceElement: HTMLSpanElement;
    private readonly originalPriceElement: HTMLSpanElement;
    private readonly benefitTitleElement: HTMLSpanElement;
    private readonly benefitTextElement: HTMLSpanElement;
    private readonly statusElement: HTMLSpanElement;
    private readonly extraElement: HTMLSpanElement;
    private readonly toggleButton: Button;
    private readonly shownIcon: SVGElement;
    private readonly hiddenIcon: SVGElement;
    private overlayVisible = true;
    private currentProduct: ProductOverlayInfo | null = null;

    constructor(events: Events, args = {}) {
        super({
            ...args,
            id: 'mobile-product-overlay'
        });

        ['pointerdown', 'pointerup', 'pointermove', 'wheel', 'dblclick', 'click'].forEach((eventName) => {
            this.dom.addEventListener(eventName, (event: Event) => {
                event.stopPropagation();
            });
        });

        this.thumbWrap = document.createElement('div');
        this.thumbWrap.className = 'mobile-product-thumb-wrap';

        this.thumbCard = document.createElement('div');
        this.thumbCard.className = 'mobile-product-thumb';

        this.thumbImage = document.createElement('img');
        this.thumbImage.alt = 'Product image';
        this.thumbImage.draggable = false;
        this.thumbCard.appendChild(this.thumbImage);

        this.toggleButton = new Button({
            class: 'mobile-product-toggle'
        });
        this.toggleButton.dom.setAttribute('aria-label', 'Toggle product info');

        this.shownIcon = createSvg(shownSvg);
        this.hiddenIcon = createSvg(hiddenSvg);
        this.hiddenIcon.classList.add('pcui-hidden');
        this.toggleButton.dom.appendChild(this.shownIcon);
        this.toggleButton.dom.appendChild(this.hiddenIcon);

        this.thumbWrap.appendChild(this.thumbCard);
        this.thumbWrap.appendChild(this.toggleButton.dom);

        this.priceBar = document.createElement('div');
        this.priceBar.className = 'mobile-product-price';

        const hero = document.createElement('div');
        hero.className = 'mobile-product-hero';

        const main = document.createElement('div');
        main.className = 'mobile-product-main';

        const priceMeta = document.createElement('div');
        priceMeta.className = 'mobile-product-price-meta';

        this.labelElement = document.createElement('span');
        this.labelElement.className = 'mobile-product-label';

        this.originalPriceElement = document.createElement('span');
        this.originalPriceElement.className = 'mobile-product-original';

        this.priceElement = document.createElement('span');
        this.priceElement.className = 'mobile-product-value';

        priceMeta.appendChild(this.labelElement);
        priceMeta.appendChild(this.originalPriceElement);
        main.appendChild(priceMeta);
        main.appendChild(this.priceElement);

        const benefit = document.createElement('div');
        benefit.className = 'mobile-product-benefit';

        this.benefitTitleElement = document.createElement('span');
        this.benefitTitleElement.className = 'mobile-product-benefit-title';

        this.benefitTextElement = document.createElement('span');
        this.benefitTextElement.className = 'mobile-product-benefit-text';

        benefit.appendChild(this.benefitTitleElement);
        benefit.appendChild(this.benefitTextElement);

        hero.appendChild(main);
        hero.appendChild(benefit);

        const details = document.createElement('div');
        details.className = 'mobile-product-details';

        this.statusElement = document.createElement('span');
        this.statusElement.className = 'mobile-product-detail-pill';

        this.extraElement = document.createElement('span');
        this.extraElement.className = 'mobile-product-detail-pill mobile-product-detail-pill-accent';

        details.appendChild(this.statusElement);
        details.appendChild(this.extraElement);

        this.priceBar.appendChild(hero);
        this.priceBar.appendChild(details);

        this.dom.appendChild(this.thumbWrap);
        this.dom.appendChild(this.priceBar);

        this.toggleButton.on('click', () => {
            this.overlayVisible = !this.overlayVisible;
            this.syncVisibility();
        });

        events.on('doc.name', (name: string | null) => {
            this.setProduct(name);
        });

        this.hidden = true;
    }

    private setProduct(name: string | null) {
        const product = productOverlayMap[getModelKey(name)] ?? null;
        this.currentProduct = product;

        if (!product) {
            this.hidden = true;
            return;
        }

        this.hidden = false;
        this.thumbImage.src = product.image;
        this.labelElement.textContent = product.label ?? '\u5230\u624B\u4EF7';
        this.priceElement.textContent = `\u00A5${product.price}`;
        this.originalPriceElement.textContent = product.originalPrice ? `\u00A5${product.originalPrice}` : '';
        this.benefitTitleElement.textContent = product.benefitTitle ?? '';
        this.benefitTextElement.textContent = product.benefitText ?? '';
        this.statusElement.textContent = product.statusText ?? '';
        this.extraElement.textContent = product.extraText ?? '';

        this.syncVisibility();
    }

    private syncVisibility() {
        const hasProduct = !!this.currentProduct;
        const collapsed = !this.overlayVisible;
        const showProduct = hasProduct && !collapsed;

        this.dom.classList.toggle('is-collapsed', collapsed);
        this.thumbCard.hidden = !showProduct;
        this.priceBar.hidden = !showProduct;
        this.thumbCard.style.display = showProduct ? 'block' : 'none';
        this.priceBar.style.display = showProduct ? 'flex' : 'none';
        this.shownIcon.style.display = collapsed ? 'none' : 'block';
        this.hiddenIcon.style.display = collapsed ? 'block' : 'none';
    }
}

export { MobileProductOverlay };
